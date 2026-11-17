# SPDX-License-Identifier: LGPL-2.1-or-later
"""ane: libane for numpy, the Apple Neural Engine on Linux.

    with Model("model.hwx", "model.json") as m:
        outputs = m.run({"x": array})       # {output name: float16 array}

The metadata (a path or a dict) names the program's inputs and outputs and their shapes:

    {"format": "ane-meta-1",
     "inputs":  [{"name": "x", "shape": [16, 32, 32], "dtype": "float32"}],
     "outputs": [{"name": "y", "shape": [16, 32, 32], "dtype": "float16"}]}

An entry with an "index" key takes the buffer of that index in the program; the others take the
remaining buffers in the order of their index. The engine computes in float16: inputs of any
dtype are cast to it and outputs come back as float16 ("dtype" records the model's own). Each
buffer holds the array's prod(shape[:-1]) rows of shape[-1] values, each row at the buffer's
size / rows bytes from the last; the padding at a row's end stays zero.

A Model is one program; its run() calls are serialised, and separate Models run concurrently.
Closing a Device closes its Models first.
"""

import ctypes
import errno
import json
import math
import os
import threading
import weakref

import numpy as np

__version__ = "0.1.0"
__all__ = ["META_FORMAT", "AneError", "Device", "Model"]
META_FORMAT = "ane-meta-1"

_lib = None


def _load():
    """The C library: $ANE_LIB, a build beside this file's directory, or the system's."""
    global _lib
    if _lib:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    names = [os.environ.get("ANE_LIB"), os.path.join(here, "..", "libane.so.0"), "libane.so.0"]
    err = None
    for n in filter(None, names):
        try:
            lib = ctypes.CDLL(n)
            break
        except OSError as e:
            err = e
    else:
        raise OSError(f"libane not found (set ANE_LIB to its path): {err}")
    vp, cp, ci = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
    sz, ssz, pp = ctypes.c_size_t, ctypes.c_ssize_t, ctypes.POINTER(ctypes.c_void_p)
    for name, res, args in [
        ("ane_open", ci, [pp, cp]),
        ("ane_close", None, [vp]),
        ("ane_load", ci, [vp, cp, pp]),
        ("ane_load_mem", ci, [vp, vp, sz, pp]),
        ("ane_free", None, [vp]),
        ("ane_procs", ci, [vp]),
        ("ane_src_count", ci, [vp, ci]),
        ("ane_dst_count", ci, [vp, ci]),
        ("ane_src_size", ssz, [vp, ci, ci]),
        ("ane_dst_size", ssz, [vp, ci, ci]),
        ("ane_src_index", ci, [vp, ci, ci]),
        ("ane_dst_index", ci, [vp, ci, ci]),
        ("ane_src_ptr", vp, [vp, ci, ci]),
        ("ane_dst_ptr", vp, [vp, ci, ci]),
        ("ane_run", ci, [vp, ci, ci]),
    ]:
        f = getattr(lib, name)
        f.restype, f.argtypes = res, args
    _lib = lib
    return lib


class AneError(OSError):
    """A libane call's negative errno, as an OSError."""


def _chk(r, what):
    if r < 0:
        raise AneError(-r, f"{what}: {os.strerror(-r)}")
    return r


class Device:
    """An open apple-ane node: NODE, else $ANE_DEVICE, else the first one under /dev/accel."""

    def __init__(self, node=None):
        self._lib = _load()
        self._models = weakref.WeakSet()
        p = ctypes.c_void_p()
        _chk(self._lib.ane_open(ctypes.byref(p), os.fsencode(node) if node else None), "ane_open")
        self._p = p

    def close(self):
        p, self._p = getattr(self, "_p", None), None
        if not p:
            return
        for m in list(self._models):
            m.close()
        self._lib.ane_close(p)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class _Buf:
    """One input or output: its mapping as rows of float16."""

    def __init__(self, spec, ptr, size):
        self.name = spec["name"]
        self.shape = tuple(int(d) for d in spec["shape"])
        if not self.shape or min(self.shape) < 1:
            raise ValueError(f"{self.name}: shape {self.shape} is not a shape of one or more axes")
        self.rows = math.prod(self.shape[:-1])
        self.width = self.shape[-1]
        if size % self.rows or size // self.rows < 2 * self.width:
            raise ValueError(
                f"{self.name}: a {size}-byte buffer is not {self.rows} rows of {self.width} halves"
            )
        stride = size // self.rows
        raw = np.ctypeslib.as_array((ctypes.c_uint8 * size).from_address(ptr))
        self.view = raw.reshape(self.rows, stride)[:, : 2 * self.width].view(np.float16)


class Model:
    """A program (HWX, a path) with its metadata (META, a path or a dict).

    It runs on DEVICE, or on a Device of its own that closes with it.
    """

    def __init__(self, hwx, meta, device=None, procedure=0):
        self._lock = threading.Lock()
        self._nn = None
        self._own = False
        self._dev = None
        self.inputs = self.outputs = ()
        if isinstance(meta, (str, os.PathLike)):
            with open(meta) as f:
                meta = json.load(f)
        if meta.get("format", META_FORMAT) != META_FORMAT:
            raise ValueError(f"metadata format {meta['format']!r}, not {META_FORMAT}")
        self._lib = _load()
        self._proc = procedure
        self._own = device is None
        self._dev = Device() if device is None else device
        if not self._dev._p:
            raise ValueError("the device is closed")
        nn = ctypes.c_void_p()
        _chk(
            self._lib.ane_load(self._dev._p, os.fsencode(hwx), ctypes.byref(nn)), f"ane_load {hwx}"
        )
        self._nn = nn
        self._dev._models.add(self)
        try:
            self.inputs = self._bind(meta["inputs"], out=False)
            self.outputs = self._bind(meta["outputs"], out=True)
        except BaseException:
            self.close()
            raise

    def _bind(self, specs, out):
        lib, nn, pr = self._lib, self._nn, self._proc
        side = "dst" if out else "src"
        count, size, index, ptr = (
            getattr(lib, f"ane_{side}_{f}") for f in ("count", "size", "index", "ptr")
        )
        kind = "outputs" if out else "inputs"
        n = _chk(count(nn, pr), f"the {kind}' count")
        if n != len(specs):
            raise ValueError(f"the program has {n} {kind}, the metadata {len(specs)}")
        # libane numbers them in the order of their index in the program
        at = {_chk(index(nn, pr, i), "a buffer's index"): i for i in range(n)}
        named = [s["index"] for s in specs if "index" in s]
        for x in named:
            if x not in at:
                raise ValueError(f"the program has none of its {kind} at index {x}")
        if len(set(named)) != len(named):
            raise ValueError(f"the metadata names an index of its {kind} twice")
        rest = [at[x] for x in sorted(at) if x not in named]
        bufs = []
        for s in specs:
            i = at[s["index"]] if "index" in s else rest.pop(0)
            p = ptr(nn, pr, i)
            if not p:
                raise AneError(errno.EIO, f"{s['name']}: no mapping")
            bufs.append(_Buf(s, p, _chk(size(nn, pr, i), "a buffer's size")))
        return tuple(bufs)

    def run(self, feeds, timeout_ms=0):
        """FEEDS: {input name: array of its shape}. Returns {output name: float16 array}."""
        with self._lock:
            if not self._nn:
                raise ValueError("the model is closed")
            unknown = set(feeds) - {b.name for b in self.inputs}
            if unknown:
                raise ValueError(f"the model has no input {sorted(unknown)[0]!r}")
            for b in self.inputs:
                a = np.asarray(feeds[b.name])
                if a.shape != b.shape:
                    raise ValueError(f"{b.name}: shape {a.shape}, the model takes {b.shape}")
                b.view[:] = a.reshape(b.rows, b.width)
            _chk(self._lib.ane_run(self._nn, self._proc, timeout_ms), "ane_run")
            return {b.name: b.view.copy().reshape(b.shape) for b in self.outputs}

    def close(self):
        """Unload the program, and close the device if it is the Model's own. Idempotent."""
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            self.inputs = self.outputs = ()
            if self._nn:
                self._lib.ane_free(self._nn)
                self._nn = None
        if self._own and self._dev:
            dev, self._dev = self._dev, None
            dev.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
