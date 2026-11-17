# libane

The Apple Neural Engine (ANE) on Linux: a small C library and a Python (numpy) wrapper over the `apple-ane` accel driver. It loads a program that Apple's compiler built for the engine, maps the program's buffers and runs it.

Status (0.1.0, 2026-11-17): tested on an M3 Max, the only engine the driver supports so far (H15). A convolution and three small networks give the same outputs as CoreML on the engine under macOS, value for value, as root and as an ordinary user.

## Requirements

- A kernel with the `apple-ane` driver, which makes `/dev/accel/accelN`. The driver is not upstream: it is in my kernel fork ([hitori-chan/linux](https://github.com/hitori-chan/linux), branch `m3-wip`) and in `linux-asahi` from [my package repository](https://github.com/hitori-chan/PKGBUILDs) from r218, with a device tree that gives the engine its own IOMMUs.
- For an ordinary user: the driver's `user_programs` parameter (on by default), which lets a user load programs because the engine's memory access is confined to the caller's own buffers; and read-write access to the node (systemd's udev rules make accel nodes `0666`). Without the confinement, loading a program needs `CAP_SYS_ADMIN`.
- A program: a `.hwx`, built on macOS by Apple's compiler from a CoreML model. There is no open compiler. `anehwx` ([below](#programs-anehwx)) compiles a model and writes its `.hwx`.
- To build: a C compiler, `make` and the kernel's uAPI headers (`linux-api-headers`). The Python wrapper needs numpy.

## Build

```
$ make
# make install PREFIX=/usr
```

This installs `libane.so.0`, `libane.a`, the header `ane.h` and `libane.pc`. Copy `python/ane.py` onto Python's path for the wrapper. The Arch packages are `libane` and `python-ane`.

## C

```c
#include <ane.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
	struct ane *dev;
	struct ane_nn *nn;
	int r;

	r = ane_open(&dev, NULL);
	if (r) {
		fprintf(stderr, "ane_open: %s\n", strerror(-r));
		return 1;
	}
	r = ane_load(dev, "model.hwx", &nn);
	if (r) {
		fprintf(stderr, "ane_load: %s\n", strerror(-r));
		ane_close(dev);
		return 1;
	}
	printf("%d inputs, %d outputs\n", ane_src_count(nn, 0), ane_dst_count(nn, 0));
	/* write input 0 in the engine's layout (below), run procedure 0, read output 0 */
	memset(ane_src_ptr(nn, 0, 0), 0, ane_src_size(nn, 0, 0));
	r = ane_run(nn, 0, 0);
	if (!r) {
		const unsigned short *y = ane_dst_ptr(nn, 0, 0);

		printf("y[0] = 0x%04x (float16)\n", y[0]);
	}
	ane_free(nn);
	ane_close(dev);
	return r ? 1 : 0;
}
```

```
$ cc example.c $(pkg-config --cflags --libs libane)
```

| Call | What it does |
|---|---|
| `ane_open(&dev, node)` | opens `node`, or with `NULL` `$ANE_DEVICE`, or else the first `apple-ane` node |
| `ane_load(dev, path, &nn)`, `ane_load_mem(dev, buf, size, &nn)` | loads a `.hwx` |
| `ane_procs(nn)` | its procedure count |
| `ane_src_count`, `ane_dst_count` | a procedure's inputs and outputs |
| `ane_src_size`, `ane_dst_size`, `ane_src_index`, `ane_dst_index` | a buffer's bytes and its index in the program |
| `ane_src_ptr`, `ane_dst_ptr` | a buffer's mapping |
| `ane_send`, `ane_read` | a copy into an input, out of an output |
| `ane_run(nn, proc, timeout_ms)` | runs a procedure once and waits for it |
| `ane_free(nn)`, `ane_close(dev)` | unloads, closes |

The inputs and outputs of a procedure are numbered from 0 in the order of their index in the program. The library maps every buffer a procedure binds, intermediates included, at the procedure's first use. Errors are negative errno values; `ane.h` lists them and states what may be shared between threads.

## Python

```python
import numpy as np
import ane

with ane.Model("model.hwx", "model.json") as m:
    y = m.run({"x": np.zeros((16, 32, 32), np.float32)})["y"]   # a float16 array
```

`Model` takes the program and its metadata, a JSON file or a dict that names the inputs and outputs and gives their shapes:

```json
{"format": "ane-meta-1",
 "inputs":  [{"name": "x", "shape": [16, 32, 32], "dtype": "float32"}],
 "outputs": [{"name": "y", "shape": [16, 32, 32], "dtype": "float16"}]}
```

The names pair with the procedure's inputs and outputs in the order of their index in the program, or by an `"index"` key giving that index. The shapes are the CoreML model's. `anehwx` writes this file beside the `.hwx` (its metadata output is not yet tested); it can be written by hand.

Several `Model`s can share one `ane.Device()`. A `Model`'s runs are serialised; separate `Model`s run concurrently. `$ANE_LIB` names the shared library when it is not installed or beside `python/`.

## Programs: anehwx

`macos/anehwx.m` compiles a CoreML model for the engine on macOS, as an ordinary user, with no change to the system:

```
% clang -fobjc-arc -O2 -o anehwx anehwx.m -framework Foundation -framework CoreML
% ./anehwx model.mlmodel h15g out
```

`out/` then holds `model.hwx`, its metadata `model.json`, and CoreML's outputs on the CPU and on the engine for one fixed input (`in.bin`, `out-cpu.bin`, `out-ane.bin`, float32) to compare libane's with. `h15g` is the M3 Max's engine. The compiler is one of macOS's private frameworks; the programs tested here come from macOS 27.2's.

## The engine's buffers

The engine computes in float16. An input or output of shape `(…, w)` is `prod(shape[:-1])` rows of `w` float16 values, each row starting `size / rows` bytes after the last; the rest of a row is padding, zero on input. The Python wrapper does this layout; a C caller does it itself. It is verified for the 3-D shapes of convolutional models; other ranks and tilings are not.

## Tests and lint

```
$ make check MODELS="conv16 net3"
$ make check SAN=1 MODELS=conv16
$ make lint
```

`make check` runs the C tests (`test/test.c`) and the Python tests (`test/test_ane.py`). A model directory holds a `model.hwx` and its `model.json`. Without an `apple-ane` node, or without models, only the checks that need neither run. With them, each program is checked for:

- the API's contracts and errors, including refused programs;
- reruns that match bit for bit;
- recovery from a run that outlasts its timeout;
- load-run-free cycles that leak no descriptor or memory;
- runs from several threads at once.

The Python tests use the first model. `SAN=1` builds the C tests with AddressSanitizer and UBSan. `make lint` runs clang-format, clang-tidy (`.clang-tidy`), `ruff format` and `ruff check` (`ruff.toml`).

## About

AI agents write libane and the kernel driver under it, at my direction; I test them on my M3 Max. Asahi Linux does not accept AI-written contributions, which I respect: this work stays in my own repositories and forks, and nothing of it is sent upstream.

## Licence

LGPL-2.1-or-later ([LICENSE](LICENSE)), `anehwx` included, so programs under other licences can link the library. `apple_ane_accel.h` is a copy of the driver's uAPI header (GPL-2.0-only WITH Linux-syscall-note). The function names follow those of [ane](https://github.com/eiln/ane), the library for the M1's engine; the code is new.
