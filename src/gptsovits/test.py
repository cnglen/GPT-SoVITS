#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""

import os

from gptsovits.cli import ASRWorker, SliceWorker, Uvr5Worker


def test_uvr5():
    for model_name in [
        # "onnx_dereverb_By_FoxJoy",
        # "model_bs_roformer_ep_317_sdr_12.9755",
        "HP2_all_vocals",
        # "HP5_only_main_vocal",
        # "VR-DeEchoAggressive",
        # "VR-DeEchoDeReverb",
        # "VR-DeEchoNormal",
    ]:
        worker = Uvr5Worker(model_name=model_name)
        worker.run(
            d_input=os.path.expanduser("~/data/test_uvrt"),
            d_output_vocal=os.path.expanduser(f"~/data/test_uvrt/vocal_{model_name}"),
            d_output_other=os.path.expanduser(f"~/data/test_uvrt/other_{model_name}"),
        )


def test_slice():
    worker = SliceWorker()
    worker.run(
        d_input=os.path.expanduser("~/data/test_uvrt/vocal_HP5_only_main_vocal"),
        d_output=os.path.expanduser("~/data/test_uvrt/vocal_HP5_only_main_vocal_split"),
    )


def test_asr():
    worker = ASRWorker()
    worker.run(
        d_input=os.path.expanduser("~/data/test_uvrt/vocal_HP5_only_main_vocal_split"),
        d_output=os.path.expanduser("~/data/test_uvrt/vocal_HP5_only_main_vocal_split/asr"),
    )


def main():
    uvr5_model_name = "HP5_only_main_vocal"
    uvr5_worker = Uvr5Worker(model_name=uvr5_model_name)
    uvr5_worker.run(
        d_input=os.path.expanduser("~/data/test_uvrt"),
        d_output_vocal=os.path.expanduser(f"~/data/test_uvrt/vocal_{uvr5_model_name}"),
        d_output_other=os.path.expanduser(f"~/data/test_uvrt/other_{uvr5_model_name}"),
    )

    slice_worker = SliceWorker()
    slice_worker.run(
        d_input=os.path.expanduser("~/data/test_uvrt/vocal_HP5_only_main_vocal"),
        d_output=os.path.expanduser("~/data/test_uvrt/vocal_HP5_only_main_vocal_split"),
    )


if __name__ == "__main__":
    # test_slice()
    test_asr()
