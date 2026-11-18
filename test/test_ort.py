# SPDX-License-Identifier: LGPL-2.1-or-later
"""The ONNX Runtime EP's tests: python3 test/test_ort.py [DIR...] (make check-ort).

A DIR is anehwx's output: model.hwx, model.json, and for one input and one output in.bin and
out-ane.bin (CoreML's engine output, float32), which the EP's must equal; the tests that run a
model skip a DIR without them. $ANE_ORT_EP names the EP's library (default
ort/libonnxruntime_ane.so). Each DIR becomes models through ort/ane-epctx.py, the program beside
the model and in it, and broken ones the EP must refuse. Needs onnxruntime (1.24 or later) and
onnx; without them, the EP or an apple-ane node, the tests skip.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = ort = None

HERE = os.path.dirname(os.path.abspath(__file__))
EPCTX = os.path.join(HERE, "..", "ort", "ane-epctx.py")
EP = os.environ.get("ANE_ORT_EP") or os.path.join(HERE, "..", "ort", "libonnxruntime_ane.so")
NAME = "AneExecutionProvider"
DIRS = []


def epctx(src, out, *flags):
    subprocess.run([sys.executable, EPCTX, src, out, *flags], check=True, capture_output=True)
    return out


class Ort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ort is None:
            raise unittest.SkipTest("no onnxruntime or onnx")
        if not hasattr(ort, "register_execution_provider_library"):
            raise unittest.SkipTest(f"onnxruntime {ort.__version__} has no plugin EPs")
        if not os.path.exists(EP):
            raise unittest.SkipTest(f"no {EP} (make ort)")
        ort.register_execution_provider_library("ane", EP)
        cls.devices = [d for d in ort.get_ep_devices() if d.ep_name == NAME]
        cls.tmp = tempfile.mkdtemp(prefix="ane-ort-")

    @classmethod
    def tearDownClass(cls):
        ort.unregister_execution_provider_library("ane")
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.devices:
            self.skipTest("the EP offers no device: no usable apple-ane node")

    def session(self, model, **config):
        so = ort.SessionOptions()
        for k, v in config.items():
            so.add_session_config_entry(k, v)
        so.add_provider_for_devices(self.devices, {})
        return ort.InferenceSession(model, sess_options=so)

    def test_device(self):
        d = self.devices[0]
        self.assertEqual(len(self.devices), 1)
        self.assertEqual(d.device.type, ort.OrtHardwareDeviceType.NPU)
        self.assertEqual(d.ep_vendor, "libane")

    def each_dir(self, runs=True):
        """Each DIR's source, copy and metadata; with RUNS those with a run's sidecars only."""
        if not DIRS:
            self.skipTest("no DIR given")
        dirs = []
        for src in DIRS:
            with open(os.path.join(src, "model.json")) as f:
                meta = json.load(f)
            if runs and not runnable(src, meta):
                print(
                    f"{src}: no in.bin and out-ane.bin for one input and one output",
                    file=sys.stderr,
                )
                continue
            dirs.append((src, meta))
        if not dirs:
            self.skipTest("no DIR with in.bin and out-ane.bin")
        for src, meta in dirs:
            # a copy the model sits beside, with the program
            dst = os.path.join(self.tmp, os.path.basename(os.path.normpath(src)))
            os.makedirs(dst, exist_ok=True)
            for f in ("model.hwx", "model.json"):
                shutil.copy(os.path.join(src, f), dst)
            yield src, dst, meta

    @staticmethod
    def io(src, meta, half=False):
        (i,), (o,) = meta["inputs"], meta["outputs"]
        x = np.fromfile(os.path.join(src, "in.bin"), np.float32).reshape(i["shape"])
        ref = np.fromfile(os.path.join(src, "out-ane.bin"), np.float32).reshape(o["shape"])
        return i["name"], (x.astype(np.float16) if half else x), ref

    def check(self, sess, src, meta, half=False):
        self.assertIn(NAME, sess.get_providers())
        name, x, ref = self.io(src, meta, half)
        (y,) = sess.run(None, {name: x})
        self.assertEqual(y.dtype, np.float16 if half else np.float32)
        np.testing.assert_array_equal(y.astype(np.float32), ref)

    def test_beside(self):
        for src, dst, meta in self.each_dir():
            with self.subTest(src):
                m = epctx(dst, os.path.join(dst, "beside.onnx"))
                self.check(self.session(m), src, meta)

    def test_embedded(self):
        for src, dst, meta in self.each_dir():
            with self.subTest(src):
                m = epctx(dst, os.path.join(self.tmp, "embedded.onnx"), "--embed")
                self.check(self.session(m), src, meta)
                with open(m, "rb") as f:  # from memory too
                    self.check(self.session(f.read()), src, meta)

    def test_float16(self):
        for src, dst, meta in self.each_dir():
            with self.subTest(src):
                m = epctx(dst, os.path.join(dst, "half.onnx"), "--float16")
                self.check(self.session(m), src, meta, half=True)

    def test_from_memory_by_path(self):
        for src, dst, meta in self.each_dir():
            with self.subTest(src):
                m = epctx(dst, os.path.join(dst, "mem.onnx"))
                with open(m, "rb") as f:
                    data = f.read()
                with self.assertRaisesRegex(Exception, "ep.context_file_path"):
                    self.session(data)
                self.check(self.session(data, **{"ep.context_file_path": m}), src, meta)

    def broken(self, dst, name, edit, where=None):
        """A model of DST's program beside it, its node changed by EDIT, saved in WHERE (DST)."""
        m = onnx.load(epctx(dst, os.path.join(dst, name)))
        edit(m.graph.node[0])
        path = os.path.join(where or dst, name)
        onnx.save(m, path)
        return path

    def refused(self, path, pattern, **config):
        """No session, and the EP's error matches PATTERN.

        ONNX Runtime's Python prints the error of an EP that failed to be made ("EP Error") and
        retries without it, which then fails on its own.
        """
        out = StringIO()
        try:
            with redirect_stdout(out):
                self.session(path, **config)
        except Exception as e:
            text = f"{e}\n{out.getvalue()}"
        else:
            self.fail(f"{path}: a session was made")
        self.assertRegex(text, pattern)

    def test_outside_refused(self):
        for _, dst, _ in self.each_dir(runs=False):
            self.refused(
                self.broken(
                    dst, "out.onnx", lambda n: set_attr(n, "ep_cache_context", b"../model.hwx")
                ),
                "leaves the model's directory",
            )
            return

    def test_symlink_out_refused(self):
        """A symlink in the model's directory to a program outside it is not followed."""
        for _, dst, _ in self.each_dir(runs=False):
            inner = os.path.join(dst, "inner")
            os.makedirs(inner, exist_ok=True)
            link = os.path.join(inner, "link.hwx")
            if not os.path.lexists(link):
                os.symlink(os.path.join("..", "model.hwx"), link)
            path = self.broken(
                dst, "sym.onnx", lambda n: set_attr(n, "ep_cache_context", b"link.hwx"), inner
            )
            self.refused(path, "leaves the model's directory")
            return

    def test_fifo_refused(self):
        """A FIFO named as the program is refused at once, not waited on."""
        for _, dst, _ in self.each_dir(runs=False):
            fifo = os.path.join(dst, "fifo.hwx")
            if not os.path.exists(fifo):
                os.mkfifo(fifo)
            path = self.broken(
                dst, "fifo.onnx", lambda n: set_attr(n, "ep_cache_context", b"fifo.hwx")
            )
            self.refused(path, "not a program file")
            return

    def test_index_refused(self):
        """An input's buffer index the program has not."""
        for _, dst, _ in self.each_dir(runs=False):
            path = self.broken(
                dst,
                "idx.onnx",
                lambda n: set_ints(n, "ane_src_index", [99]),
            )
            self.refused(path, "no buffer at index 99")
            return

    def test_count_refused(self):
        """A node with more inputs than the program's: one tensor fed twice included."""
        for _, dst, _ in self.each_dir(runs=False):
            path = self.broken(dst, "twice.onnx", lambda n: n.input.append(n.input[0]))
            self.refused(path, r"the program has \d+ inputs, the node \d+")
            return

    def test_omitted_refused(self):
        """A node whose input is omitted (an empty name) fails cleanly, with no crash."""
        for _, dst, _ in self.each_dir(runs=False):
            path = self.broken(dst, "omit.onnx", lambda n: n.input.__setitem__(0, ""))
            self.refused(path, "omits its input 0")
            return

    def test_options_refused(self):
        """A timeout that is no count of ms, and a context model asked of a compiled one."""
        for _, dst, _ in self.each_dir(runs=False):
            m = epctx(dst, os.path.join(dst, "opts.onnx"))
            for bad in ("abc", "0", "-5", ""):
                self.refused(m, "timeout_ms", **{f"ep.{NAME}.timeout_ms": bad})
            self.refused(m, "compiled already", **{"ep.context_enable": "1"})
            return

    def test_cycles(self):
        """Sessions made and dropped leak no descriptor and little memory."""
        for src, dst, meta in self.each_dir():
            m = epctx(dst, os.path.join(dst, "cycles.onnx"))
            name, x, ref = self.io(src, meta)
            for _ in range(3):  # ONNX Runtime's own first-use allocations
                self.session(m).run(None, {name: x})
            gc.collect()
            fds, rss = len(os.listdir("/proc/self/fd")), rss_kb()
            for _ in range(50):
                (y,) = self.session(m).run(None, {name: x})
                np.testing.assert_array_equal(y, ref)
            gc.collect()
            self.assertEqual(len(os.listdir("/proc/self/fd")), fds)
            if not os.environ.get("ANE_ORT_SAN"):  # ASan's quarantine holds what is freed
                self.assertLess(rss_kb() - rss, 32 * 1024)  # 50 sessions: well under 32 MiB
            return

    def test_threads(self):
        for src, dst, meta in self.each_dir():
            sess = self.session(epctx(dst, os.path.join(dst, "threads.onnx")))
            name, x, ref = self.io(src, meta)
            bad = []
            ts = [threading.Thread(target=runs, args=(sess, name, x, ref, bad)) for _ in range(4)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            self.assertEqual(bad, [])
            return


def rss_kb():
    with open("/proc/self/status") as f:
        return next(int(ln.split()[1]) for ln in f if ln.startswith("VmRSS:"))


def runs(sess, name, x, ref, bad):
    try:
        for _ in range(10):
            (y,) = sess.run(None, {name: x})
            if not np.array_equal(y, ref):
                bad.append("a wrong output")
    except Exception as e:  # a thread's failure is the test's
        bad.append(repr(e))


def runnable(src, meta):
    one = len(meta["inputs"]) == 1 and len(meta["outputs"]) == 1
    return one and all(os.path.exists(os.path.join(src, f)) for f in ("in.bin", "out-ane.bin"))


def set_ints(node, name, values):
    for a in [a for a in node.attribute if a.name == name]:
        node.attribute.remove(a)
    node.attribute.append(onnx.helper.make_attribute(name, values))


def set_attr(node, name, value):
    for a in node.attribute:
        if a.name == name:
            a.s = value


if __name__ == "__main__":
    DIRS[:] = [d for d in sys.argv[1:] if not d.startswith("-")]
    unittest.main(argv=[sys.argv[0]] + [d for d in sys.argv[1:] if d.startswith("-")])
