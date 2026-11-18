#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""ane-epctx.py DIR [OUT] [--embed] [--float16]: an ONNX model that runs anehwx's program on the
Neural Engine through ONNX Runtime's ane EP (libonnxruntime_ane.so).

DIR holds model.hwx and its model.json (anehwx's). OUT (default DIR/model_ctx.onnx) gets one
EPContext node (com.microsoft) with source "ane": its inputs and outputs are model.json's, typed
float32 (float16 with --float16, or where model.json says float16), with their shapes. The
program is named by its path relative to OUT's directory, which must hold it (embed_mode 0), or
with --embed its bytes are in the node (embed_mode 1). Entries of model.json with an "index" give
the node's ane_src_index and ane_dst_index attributes; then every entry needs one.
"""

import argparse
import json
import os
import sys

import onnx
from onnx import TensorProto, helper

META_FORMAT = "ane-meta-1"
# ONNX Runtime 1.29 reads IR versions up to 13 (onnx 1.23 writes 14 by default)
IR_VERSION = 10


def value(spec, half):
    # model.json's dtype is CoreML's interface; the engine's buffer is float16 whatever it is
    t = TensorProto.FLOAT16 if half or spec.get("dtype") == "float16" else TensorProto.FLOAT
    return helper.make_tensor_value_info(spec["name"], t, [int(d) for d in spec["shape"]])


def indices(specs, kind):
    have = ["index" in s for s in specs]
    if not any(have):
        return None
    if not all(have):
        sys.exit(f"ane-epctx: some {kind} of model.json have an index and some do not")
    return [int(s["index"]) for s in specs]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--embed", action="store_true", help="the program's bytes in the node")
    ap.add_argument("--float16", action="store_true", help="float16 inputs and outputs")
    a = ap.parse_args()
    hwx = os.path.join(a.dir, "model.hwx")
    with open(os.path.join(a.dir, "model.json")) as f:
        meta = json.load(f)
    if meta.get("format", META_FORMAT) != META_FORMAT:
        sys.exit(f"ane-epctx: metadata format {meta['format']!r}, not {META_FORMAT}")
    out = a.out or os.path.join(a.dir, "model_ctx.onnx")
    if a.embed:
        with open(hwx, "rb") as f:
            ctx = f.read()
    else:
        ctx = os.path.relpath(os.path.abspath(hwx), os.path.dirname(os.path.abspath(out)))
        if ctx == os.pardir or ctx.startswith(os.pardir + os.sep):
            sys.exit("ane-epctx: OUT's directory must hold model.hwx (or use --embed)")
    attrs = {
        "main_context": 1,
        "embed_mode": 1 if a.embed else 0,
        "ep_cache_context": ctx,
        "source": "ane",
        "partition_name": "ane_0",
    }
    for kind, key in (("inputs", "ane_src_index"), ("outputs", "ane_dst_index")):
        x = indices(meta[kind], kind)
        if x is not None:
            attrs[key] = x
    node = helper.make_node(
        "EPContext",
        [s["name"] for s in meta["inputs"]],
        [s["name"] for s in meta["outputs"]],
        name="ane_0",
        domain="com.microsoft",
        **attrs,
    )
    graph = helper.make_graph(
        [node],
        "ane",
        [value(s, a.float16) for s in meta["inputs"]],
        [value(s, a.float16) for s in meta["outputs"]],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        producer_name="ane-epctx",
    )
    model.ir_version = IR_VERSION
    onnx.save(model, out)
    print(out)


if __name__ == "__main__":
    main()
