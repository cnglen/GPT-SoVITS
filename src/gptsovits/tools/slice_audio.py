import os
import sys
import traceback
from typing import List

import numpy as np
from gptsovits.tools.my_utils import load_audio
from gptsovits.tools.slicer2 import Slicer
from scipy.io import wavfile


def slice(
    inp: str | List[str],
    opt_root: str = "/tmp/audio_slice_output",
    threshold: int = -34,
    min_length: int = 4000,
    min_interval: int = 10,
    hop_size: int = 10,
    max_sil_kept: int = 500,
    _max: float = 0.9,
    alpha: float = 0.25,
    i_part: int = 0,
    all_part: int | None = None,
):
    """
    将输入`inp`进行音频切分

    Args:
      inp: 输入文件或文件夹
      opt_root: 输出文件夹
      threshold: 音量小于这个值视作静音的备选切割点
      min_length: 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
      min_interval: # 最短切割间隔
      hop_size: 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
      max_sil_kept: 切完后静音最多留多长
      _max:
      alpha:
      i_part:
      all_part:
    """
    os.makedirs(opt_root, exist_ok=True)
    if os.path.isfile(inp):
        f_inputs = [inp]
    elif os.path.isdir(inp):
        f_inputs = [os.path.join(inp, name) for name in sorted(list(os.listdir(inp)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=int(min_interval),  # 最短切割间隔
        hop_size=int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max = float(_max)
    alpha = float(alpha)
    if all_part is None:
        all_part = len(f_inputs)
    for inp_path in f_inputs[int(i_part) :: int(all_part)]:
        # print(inp_path)
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            # print(audio.shape)
            for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                    32000,
                    # chunk.astype(np.float32),
                    (chunk * 32767).astype(np.int16),
                )
        except:
            print(inp_path, "->fail->", traceback.format_exc())
    return "执行完毕，请检查输出文件"


print(slice(*sys.argv[1:]))
