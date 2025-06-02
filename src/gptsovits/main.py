#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""

__author__ = ""
__email__ = ""
__status__ = "DEV"

import os
import unittest
from typing import Dict, List, Tuple

from pylightkit.utils import full_path

from gptsovits.cli import ASRWorker, SliceWorker, Uvr5Worker

DATA_ROOT = full_path("~/data")


class PipeLine:
    def __init__(
        self,
        d_input: str,
        uvr5_model_name: str = "HP2_all_vocals",
    ):
        self.slice_worker = SliceWorker(threshold=-40, min_length=4000, min_interval=300, hop_size=10)
        self.uvr5_worker = Uvr5Worker(model_name=uvr5_model_name)
        self.asr_worker = ASRWorker()
        self.d_input_uvr5 = full_path(d_input)
        self.d_output_vocal = os.path.join(self.d_input_uvr5, "vocal")
        self.d_output_other = os.path.join(self.d_input_uvr5, "other")

        self.d_input_slice, self.d_output_slice = self.d_output_vocal, os.path.join(self.d_input_uvr5, "slice")
        self.d_output_asr = os.path.join(self.d_input_uvr5, "asr")

    def run(self):
        self.uvr5_worker.run(
            d_input=self.d_input_uvr5,
            d_output_vocal=self.d_output_vocal,
            d_output_other=self.d_output_other,
        )

        self.slice_worker.run(
            d_input=self.d_input_slice,
            d_output=self.d_output_slice,
        )

        self.asr_worker.run(d_input=self.d_output_slice, d_output=self.d_output_asr)


def main():
    pipeline = PipeLine(d_input="~/data/raw/")
    pipeline.run()


if __name__ == "__main__":
    main()
