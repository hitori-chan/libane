# SPDX-License-Identifier: LGPL-2.1-or-later
"""The Python wrapper's tests: python3 test/test_ane.py [MODEL_DIR...].

A MODEL_DIR holds model.hwx and its model.json; the first one given is used. Without an apple-ane
node, or without a MODEL_DIR, only the checks that need neither run.
"""

import copy
import errno
import json
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import ane

MODEL = None


def have_device():
    try:
        ane.Device().close()
    except OSError:
        return False
    return True


def fds():
    return len(os.listdir("/proc/self/fd"))


class NoDevice(unittest.TestCase):
    def test_not_ane(self):
        with self.assertRaises(ane.AneError) as c:
            ane.Device("/dev/null")
        self.assertEqual(c.exception.errno, errno.ENODEV)

    def test_missing(self):
        with self.assertRaises(ane.AneError) as c:
            ane.Device("/nonexistent/accel0")
        self.assertEqual(c.exception.errno, errno.ENOENT)

    def test_format(self):
        with self.assertRaisesRegex(ValueError, "format"):
            ane.Model("/nonexistent.hwx", {"format": "other", "inputs": [], "outputs": []})


class WithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL:
            raise unittest.SkipTest("no MODEL_DIR given")
        if not have_device():
            raise unittest.SkipTest("no usable apple-ane node")
        cls.hwx = os.path.join(MODEL, "model.hwx")
        with open(os.path.join(MODEL, "model.json")) as f:
            cls.meta = json.load(f)
        rng = np.random.default_rng(1)
        cls.feeds = {
            s["name"]: rng.uniform(-1, 1, s["shape"]).astype(np.float32) for s in cls.meta["inputs"]
        }
        with ane.Model(cls.hwx, cls.meta) as m:
            cls.ref = m.run(cls.feeds)

    def same(self, out):
        self.assertEqual(out.keys(), self.ref.keys())
        for k, v in out.items():
            self.assertEqual(v.dtype, np.float16)
            shape = next(s["shape"] for s in self.meta["outputs"] if s["name"] == k)
            self.assertEqual(v.shape, tuple(shape))
            np.testing.assert_array_equal(v, self.ref[k])

    def test_rerun(self):
        with ane.Model(self.hwx, self.meta) as m:
            self.same(m.run(self.feeds))
            self.same(m.run(self.feeds))

    def test_meta_path(self):
        with ane.Model(self.hwx, os.path.join(MODEL, "model.json")) as m:
            self.same(m.run(self.feeds))

    def test_outputs_follow_inputs(self):
        other = {k: -v for k, v in self.feeds.items()}
        with ane.Model(self.hwx, self.meta) as m:
            out = m.run(other)
        self.assertFalse(all(np.array_equal(out[k], self.ref[k]) for k in out))

    def test_feed_errors(self):
        with ane.Model(self.hwx, self.meta) as m:
            name = self.meta["inputs"][0]["name"]
            bad = dict(self.feeds)
            bad[name] = np.zeros((1, *self.feeds[name].shape), np.float32)
            with self.assertRaisesRegex(ValueError, "shape"):
                m.run(bad)
            with self.assertRaisesRegex(ValueError, "no input"):
                m.run({**self.feeds, "nonesuch": 0})
            with self.assertRaises(KeyError):
                m.run({})
            self.same(m.run(self.feeds))

    def test_meta_errors(self):
        n0 = fds()
        cases = []
        m = copy.deepcopy(self.meta)
        m["inputs"].append(dict(m["inputs"][0], name="extra"))
        cases.append((m, "has"))
        m = copy.deepcopy(self.meta)
        m["outputs"][0]["index"] = 10**6
        cases.append((m, "index"))
        m = copy.deepcopy(self.meta)
        m["outputs"][0]["shape"] = [3, 7, 1000003]
        cases.append((m, "rows"))
        m = copy.deepcopy(self.meta)
        m["outputs"][0]["shape"] = []
        cases.append((m, "shape"))
        for meta, msg in cases:
            with self.assertRaisesRegex(ValueError, msg):
                ane.Model(self.hwx, meta)
        self.assertEqual(fds(), n0, "a failed Model leaks no descriptor")

    def test_close(self):
        m = ane.Model(self.hwx, self.meta)
        m.close()
        m.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            m.run(self.feeds)

    def test_shared_device(self):
        with ane.Device() as d:
            a, b = ane.Model(self.hwx, self.meta, d), ane.Model(self.hwx, self.meta, d)
            self.same(a.run(self.feeds))
            self.same(b.run(self.feeds))
            b.close()
            self.same(a.run(self.feeds))
        with self.assertRaisesRegex(ValueError, "closed"):
            a.run(self.feeds)
        with self.assertRaisesRegex(ValueError, "closed"):
            ane.Model(self.hwx, self.meta, d)

    def test_threads(self):
        errs = []

        def work(m):
            try:
                for _ in range(30):
                    out = m.run(self.feeds)
                    if not all(np.array_equal(out[k], self.ref[k]) for k in out):
                        errs.append("mismatch")
            except Exception as e:
                errs.append(repr(e))

        with ane.Device() as d:
            ms = [ane.Model(self.hwx, self.meta, d) for _ in range(3)]
            ts = [threading.Thread(target=work, args=(m,)) for m in (*ms, ms[0])]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        self.assertEqual(errs, [])

    def test_cycles(self):
        n0 = fds()
        for _ in range(50):
            with ane.Model(self.hwx, self.meta) as m:
                m.run(self.feeds)
        for _ in range(20):
            ane.Model(self.hwx, self.meta).run(self.feeds)  # closed by the collector
        self.assertEqual(fds(), n0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL = sys.argv.pop(1)
        del sys.argv[1:]
    unittest.main(verbosity=2)
