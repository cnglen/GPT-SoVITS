from __future__ import absolute_import, division, print_function, unicode_literals

# sys.path.append(AP_BWE_main_dir_path)
import json
import os
import sys

import torch
import torchaudio.functional as aF
from gptsovits.tools.AP_BWE_main.datasets1.dataset import amp_pha_istft, amp_pha_stft
from gptsovits.tools.AP_BWE_main.models.model import APNet_BWE_Model

AP_BWE_MAIN_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AP_BWE_main")
# from attrdict import AttrDict####will be bug in py3.10


class AP_BWE:
    def __init__(self, device, DictToAttrRecursive, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = "%s/24kto48k/g_24kto48k.zip" % (AP_BWE_MAIN_DIR_PATH)
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError
        config_file = os.path.join(os.path.split(checkpoint_file)[0], "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        # h = AttrDict(json_config)
        h = DictToAttrRecursive(json_config)
        model = APNet_BWE_Model(h).to(device)
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict["generator"])
        model.eval()
        self.device = device
        self.model = model
        self.h = h

    def to(self, *arg, **kwargs):
        self.model.to(*arg, **kwargs)
        self.device = self.model.conv_pre_mag.weight.device
        return self

    def __call__(self, audio, orig_sampling_rate):
        with torch.no_grad():
            # audio, orig_sampling_rate = torchaudio.load(inp_path)
            # audio = audio.to(self.device)
            audio = aF.resample(audio, orig_freq=orig_sampling_rate, new_freq=self.h.hr_sampling_rate)
            amp_nb, pha_nb, com_nb = amp_pha_stft(audio, self.h.n_fft, self.h.hop_size, self.h.win_size)
            amp_wb_g, pha_wb_g, com_wb_g = self.model(amp_nb, pha_nb)
            audio_hr_g = amp_pha_istft(amp_wb_g, pha_wb_g, self.h.n_fft, self.h.hop_size, self.h.win_size)
            # sf.write(opt_path, audio_hr_g.squeeze().cpu().numpy(), self.h.hr_sampling_rate, 'PCM_16')
            return audio_hr_g.squeeze().cpu().numpy(), self.h.hr_sampling_rate
