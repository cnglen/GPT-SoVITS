#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""

import os
import unittest

from pylightkit.utils import full_path

from gptsovits.cli import ASRWorker, SliceWorker, Uvr5Worker

DATA_ROOT = full_path("~/data")


class WorkerTest(unittest.TestCase):
    def setUp(self, uvr5_model_name: str = "HP2_all_vocals"):
        self.slice_worker = SliceWorker()
        self.uvr5_worker = Uvr5Worker(model_name=uvr5_model_name)
        self.asr_worker = ASRWorker()
        self.d_input_slice, self.d_output_slice = os.path.join(DATA_ROOT, "test"), os.path.join(DATA_ROOT, "test/slice")
        self.d_output_vocal = os.path.join(DATA_ROOT, "test/slice/vocal")
        self.d_output_other = os.path.join(DATA_ROOT, "test/slice/other")
        self.d_output_asr = os.path.join(DATA_ROOT, "test/asr")

    def test(self):
        self.slice_worker.run(
            d_input=self.d_input_slice,
            d_output=self.d_output_slice,
        )
        self.assertTrue(len(os.listdir(self.d_output_slice)) >= 1)
        self.uvr5_worker.run(
            d_input=self.d_output_slice,
            d_output_vocal=self.d_output_vocal,
            d_output_other=self.d_output_other,
        )
        self.assertTrue(len(os.listdir(self.d_output_vocal)) >= 1)
        self.assertTrue(len(os.listdir(self.d_output_other)) >= 1)
        self.asr_worker.run(d_input=self.d_output_vocal, d_output=self.d_output_asr)
        self.assertTrue(len(os.listdir(self.d_output_asr)) >= 1)

    def tearDown(self):
        cmds = [f"rm -r -f {self.d_output_slice}", f"rm -r -f {self.d_output_vocal}", f"rm -r -f {self.d_output_other}"]
        for cmd in cmds:
            print(cmd)


if __name__ == "__main__":
    unittest.main()
