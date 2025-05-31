#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推理
"""

import argparse
import json
import logging
import os
import re
import sys
import traceback
from time import time as ttime
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from peft import LoraConfig, get_peft_model
from pylightkit.utils import get_logger
from transformers import AutoModelForMaskedLM, AutoTokenizer

from gptsovits.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from gptsovits.BigVGAN import bigvgan
from gptsovits.feature_extractor import cnhubert
from gptsovits.module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from gptsovits.module.models import Generator, SynthesizerTrn, SynthesizerTrnV3
from gptsovits.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from gptsovits.text import chinese, cleaned_text_to_sequence
from gptsovits.text.cleaner import clean_text
from gptsovits.text.LangSegmenter import LangSegmenter
from gptsovits.tools.audio_sr import AP_BWE
from gptsovits.tools.i18n.i18n import I18nAuto

PATH_SOVITS_V3 = os.path.join(os.path.dirname(__file__), "pretrained_models/s2Gv3.pth")
PATH_SOVITS_V4 = os.path.join(os.path.dirname(__file__), "pretrained_models/gsv-v4-pretrained/s2Gv4.pth")
F_WEIGHT = os.path.join(os.path.dirname(__file__), "weight.json")
print(PATH_SOVITS_V3)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class Worker:
    """
    Args:
      gpt_path: s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt/s1v3.ckpt
      sovits_path: s2G488k/s2G2333k/s2Gv3/s2Gv4
      bert_path:
      cnhubert_base_path:
    """

    def __init__(
        self,
        gpt_path: str = os.path.join(os.path.dirname(__file__), "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
        sovits_path: str = os.path.join(os.path.dirname(__file__), "pretrained_models/s2G488k.pth"),
        is_half: bool = False,
        device: Optional[str] = None,
        hz: int = 50,
        bert_path: str = os.path.join(os.path.dirname(__file__), "pretrained_models/chinese-roberta-wwm-ext-large"),
        cnhubert_base_path: str = os.path.join(os.path.dirname(__file__), "pretrained_models/chinese-hubert-base"),
        language: str = "zh_CN",
    ):
        self.logger = get_logger("inference", f_log="/tmp/inference.log", f_error="/tmp/inference.err")
        self.version = self.model_version = os.environ.get("version", "v2")
        self.logger.info("init: version={}, model_version={}".format(self.version, self.model_version))

        pretrained_sovits_name = [
            os.path.join(os.path.dirname(__file__), "pretrained_models/s2G488k.pth"),
            os.path.join(os.path.dirname(__file__), "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"),
            os.path.join(os.path.dirname(__file__), "pretrained_models/s2Gv3.pth"),
            os.path.join(os.path.dirname(__file__), "pretrained_models/gsv-v4-pretrained/s2Gv4.pth"),
        ]
        pretrained_gpt_name = [
            os.path.join(os.path.dirname(__file__), "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
            os.path.join(os.path.dirname(__file__), "pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
            os.path.join(os.path.dirname(__file__), "pretrained_models/s1v3.ckpt"),
            os.path.join(os.path.dirname(__file__), "pretrained_models/s1v3.ckpt"),
        ]

        _ = [[], []]
        for i in range(4):
            if os.path.exists(pretrained_gpt_name[i]):
                _[0].append(pretrained_gpt_name[i])
            if os.path.exists(pretrained_sovits_name[i]):
                _[-1].append(pretrained_sovits_name[i])
        pretrained_gpt_name, pretrained_sovits_name = _

        if not os.path.exists(F_WEIGHT):
            with open(F_WEIGHT, "w", encoding="utf-8") as f:
                json.dump({"GPT": {}, "SoVITS": {}}, f)

        with open(F_WEIGHT, "r", encoding="utf-8") as f:
            weight_data = f.read()
            weight_data = json.loads(weight_data)
            gpt_path = os.environ.get("gpt_path", weight_data.get("GPT", {}).get(self.version, pretrained_gpt_name))
            sovits_path = os.environ.get("sovits_path", weight_data.get("SoVITS", {}).get(self.version, pretrained_sovits_name))
            if isinstance(gpt_path, list):
                gpt_path = gpt_path[0]
            if isinstance(sovits_path, list):
                sovits_path = sovits_path[0]
            self.logger.info(f"{gpt_path}, {sovits_path}")

        self.mel_fn = lambda x: mel_spectrogram_torch(
            x,
            **{
                "n_fft": 1024,
                "win_size": 1024,
                "hop_size": 256,
                "num_mels": 100,
                "sampling_rate": 24000,
                "fmin": 0,
                "fmax": None,
                "center": False,
            },
        )
        self.mel_fn_v4 = lambda x: mel_spectrogram_torch(
            x,
            **{
                "n_fft": 1280,
                "win_size": 1280,
                "hop_size": 320,
                "num_mels": 100,
                "sampling_rate": 32000,
                "fmin": 0,
                "fmax": None,
                "center": False,
            },
        )

        self.spec_min, self.spec_max = -12, 2

        self.cache = {}
        self.i18n = I18nAuto(language=language)
        self.hz = hz
        self.is_half = is_half
        self.gpt_path = gpt_path
        self.sr_model = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # bert model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half:
            bert_model = bert_model.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.bert_model = bert_model.to(self.device)

        cnhubert.cnhubert_base_path = cnhubert_base_path
        ssl_model = cnhubert.get_model()
        if self.is_half:
            ssl_model = ssl_model.half()
        self.ssl_model = ssl_model.to(self.device)

        self.config, self.t2s_model = self.change_gpt_weights(gpt_path)
        self.hps, self.vq_model = self.change_sovits_weights(sovits_path)

        if self.model_version == "v1":
            self.dict_language = {
                self.i18n("中文"): "all_zh",  # 全部按中文识别
                self.i18n("英文"): "en",  # 全部按英文识别#######不变
                self.i18n("日文"): "all_ja",  # 全部按日文识别
                self.i18n("中英混合"): "zh",  # 按中英混合识别####不变
                self.i18n("日英混合"): "ja",  # 按日英混合识别####不变
                self.i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
            }
        else:
            self.dict_language = {
                self.i18n("中文"): "all_zh",  # 全部按中文识别
                self.i18n("英文"): "en",  # 全部按英文识别#######不变
                self.i18n("日文"): "all_ja",  # 全部按日文识别
                self.i18n("粤语"): "all_yue",  # 全部按中文识别
                self.i18n("韩文"): "all_ko",  # 全部按韩文识别
                self.i18n("中英混合"): "zh",  # 按中英混合识别####不变
                self.i18n("日英混合"): "ja",  # 按日英混合识别####不变
                self.i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
                self.i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
                self.i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
                self.i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
            }

        self.punctuation = set(["!", "?", "…", ",", ".", "-", " "])
        self.splits = {
            "，",
            "。",
            "？",
            "！",
            ",",
            ".",
            "?",
            "!",
            "~",
            ":",
            "：",
            "—",
            "…",
        }
        self.resample_transform_dict = {}

        self.bigvgan_model, self.hifigan_model = None, None
        if self.model_version == "v3":
            self.bigvgan_model = self.init_bigvgan()
        if self.model_version == "v4":
            self.hifigan_model = self.init_hifigan()

    def max_sec(self) -> int:
        return self.config["data"]["max_sec"]

    def change_gpt_weights(self, gpt_path: str) -> Tuple:
        """
        Load t2s model and update weight.json
        """
        # t2s model
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        # total = sum([param.nelement() for param in t2s_model.parameters()])
        # print("Number of parameter: %.2fM" % (total / 1e6))

        with open(F_WEIGHT) as f:
            data = f.read()
            data = json.loads(data)
            data["GPT"][self.version] = gpt_path
        with open(F_WEIGHT, "w") as f:
            f.write(json.dumps(data))

        self.gpt_path = gpt_path
        return config, t2s_model

    def change_sovits_weights(
        self,
        sovits_path: str,
    ):
        """
        Args:
          sovits_path:
        """
        version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
        self.logger.info(f"sovit_path={sovits_path}, version={version}, model_version={model_version}, if_lora_v3={if_lora_v3}")

        if model_version == "v3":
            is_exist = os.path.exists(PATH_SOVITS_V3)
        elif model_version == "v4":
            is_exist = os.path.exists(PATH_SOVITS_V4)
        elif model_version == "v2":
            is_exist = os.path.exists(PATH_SOVITS_V4)
        else:
            raise ValueError(f"Only v3/v4 supported, proviced {model_version}")

        if if_lora_v3 and (not is_exist):
            info = os.path.join(os.path.dirname(__file__), "pretrained_models/s2Gv3.pth") + self.i18n("SoVITS %s 底模缺失，无法加载相应 LoRA 权重" % model_version)
            raise FileExistsError(info)

        dict_s2 = load_sovits_new(sovits_path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            hps.model.version = "v2"  # v3model,v2sybomls
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        version = hps.model.version
        self.logger.info("sovits版本: {}".format(hps.model.version))
        if model_version not in {"v3", "v4"}:
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model,
            )
            model_version = version
        else:
            hps.model.version = model_version
            vq_model = SynthesizerTrnV3(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model,
            )
        if "pretrained" not in sovits_path:
            try:
                del vq_model.enc_q
            except Exception:
                pass
        if self.is_half:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
        vq_model.eval()

        if not if_lora_v3:
            self.logger.info("loading sovits_{}: {}".format(model_version, vq_model.load_state_dict(dict_s2["weight"], strict=False)))
        else:
            path_sovits = PATH_SOVITS_V3 if model_version == "v3" else PATH_SOVITS_V4
            self.logger.info(
                "if_lora_v3 loading sovits_{}pretrained_G".format(
                    model_version, vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False)
                ),
            )
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
            self.logger.info("loading sovits_{}_lora{}".format(model_version, lora_rank))
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
            vq_model.cfm = vq_model.cfm.merge_and_unload()
            # torch.save(vq_model.state_dict(),"merge_win.pth")
            vq_model.eval()

        with open(F_WEIGHT) as f:
            data = f.read()
            data = json.loads(data)
            data["SoVITS"][version] = sovits_path
        with open(F_WEIGHT, "w") as f:
            f.write(json.dumps(data))

        # fixme: versoin/model_version???
        self.logger.info(f"version: {self.version} -> {version}, model_version: {self.model_version} -> {model_version}")
        self.version, self.model_version = version, model_version
        self.sovits_path = sovits_path
        return hps, vq_model

    def get_bert_feature(self, text: str, word2ph) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_phones_and_bert(self, text, language: str, version: str, final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "all_zh":
                if re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "all_yue" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half else torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            self.logger.info(textlist)
            self.logger.info(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(self.dtype), norm_text

    def split(self, todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut1(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
        else:
            opts = [inp]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut2(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        # print(opts)
        if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut3(self, inp):
        inp = inp.strip("\n")
        opts = ["%s" % item for item in inp.strip("。").split("。")]
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    def cut4(self, inp):
        inp = inp.strip("\n")
        opts = re.split(r"(?<!\d)\.(?!\d)", inp.strip("."))
        opts = [item for item in opts if not set(item).issubset(self.punctuation)]
        return "\n".join(opts)

    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    def cut5(self, inp):
        inp = inp.strip("\n")
        punds = {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", ";", "：", "…"}
        mergeitems = []
        items = []

        for i, char in enumerate(inp):
            if char in punds:
                if char == "." and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                    items.append(char)
                else:
                    items.append(char)
                    mergeitems.append("".join(items))
                    items = []
            else:
                items.append(char)

        if items:
            mergeitems.append("".join(items))

        opt = [item for item in mergeitems if not set(item).issubset(punds)]
        return "\n".join(opt)

    @staticmethod
    def merge_short_text_in_array(texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def process_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(self.i18n("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    @staticmethod
    def clean_text_inf(text, language, version):
        language = language.replace("all_", "")
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def init_bigvgan(self):
        now_dir = os.getcwd()
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            "%s/gptsovits/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,),
            use_cuda_kernel=False,
        )  # if True, RuntimeError: Ninja is required to load C++ extensions
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval()
        if self.hifigan_model:
            self.hifigan_model = self.hifigan_model.cpu()
            self.hifigan_model = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if self.is_half:
            bigvgan_model = bigvgan_model.half().to(self.device)
        else:
            bigvgan_model = bigvgan_model.to(self.device)

        return bigvgan_model

    def init_hifigan(self):
        now_dir = os.getcwd()
        hifigan_model = Generator(
            initial_channel=100,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 6, 2, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 12, 4, 4, 4],
            gin_channels=0,
            is_bias=True,
        )
        hifigan_model.eval()
        hifigan_model.remove_weight_norm()
        state_dict_g = torch.load("%s/gptsovits/pretrained_models/gsv-v4-pretrained/vocoder.pth" % (now_dir,), map_location="cpu")
        self.logger.info("loading vocoder: {}".format(hifigan_model.load_state_dict(state_dict_g)))
        if self.bigvgan_model:
            self.bigvgan_model = self.bigvgan_model.cpu()
            self.bigvgan_model = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if self.is_half:
            hifigan_model = hifigan_model.half().to(self.device)
        else:
            hifigan_model = hifigan_model.to(self.device)

        return hifigan_model

    @staticmethod
    def get_spepc(hps, filename):
        # audio = load_audio(filename, int(hps.data.sampling_rate))
        audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec

    def resample(self, audio_tensor, sr0, sr1):
        key = "%s-%s" % (sr0, sr1)
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(self.device)
        return self.resample_transform_dict[key](audio_tensor)

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)  # .to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)

        return bert

    def get_tts_wav(
        self,
        ref_wav_path: str,
        prompt_text,
        prompt_language,
        text: str,
        text_language,
        how_to_cut=None,
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
        speed=1,
        if_freeze=False,
        inp_refs=None,
        sample_steps=8,
        if_sr=False,
        pause_second=0.3,
    ):
        if how_to_cut is None:
            how_to_cut = self.i18n("不切")
        t = []
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        if self.model_version in {"v3", "v4"}:
            # 开启无参考文本模式。不填参考文本亦相当于开启。v3暂不支持该模式，使用了会报错。
            ref_free = False  # s2v3暂不支持ref_free
        else:
            if_sr = False

        t0 = ttime()
        prompt_language = self.dict_language[prompt_language]
        text_language = self.dict_language[text_language]

        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in self.splits:
                prompt_text += "。" if prompt_language != "en" else "."
            self.logger.info("{}: {}".format(self.i18n("实际输入的参考文本"), prompt_text))
        text = text.strip("\n")
        # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        self.logger.info("{}: {}".format(self.i18n("实际输入的目标文本"), text))
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * pause_second),
            dtype=np.float16 if self.is_half else np.float32,
        )
        zero_wav_torch = torch.from_numpy(zero_wav)
        if self.is_half:
            zero_wav_torch = zero_wav_torch.half().to(self.device)
        else:
            zero_wav_torch = zero_wav_torch.to(self.device)
        if not ref_free:
            with torch.no_grad():
                wav16k, sr = librosa.load(ref_wav_path, sr=16000)
                if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                    logging.warning(self.i18n("参考音频在3~10秒范围外，请更换！"))
                    raise OSError(self.i18n("参考音频在3~10秒范围外，请更换！"))
                wav16k = torch.from_numpy(wav16k)
                if self.is_half:
                    wav16k = wav16k.half().to(self.device)
                else:
                    wav16k = wav16k.to(self.device)
                wav16k = torch.cat([wav16k, zero_wav_torch])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(self.device)

        t1 = ttime()
        t.append(t1 - t0)

        if how_to_cut == self.i18n("凑四句一切"):
            text = self.cut1(text)
        elif how_to_cut == self.i18n("凑50字一切"):
            text = self.cut2(text)
        elif how_to_cut == self.i18n("按中文句号。切"):
            text = self.cut3(text)
        elif how_to_cut == self.i18n("按英文句号.切"):
            text = self.cut4(text)
        elif how_to_cut == self.i18n("按标点符号切"):
            text = self.cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        self.logger.info("{}: {}".format(self.i18n("实际输入的目标文本(切句后)"), text))
        texts = text.split("\n")
        texts = self.process_text(texts)
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        ###s2v3暂不支持ref_free
        if not ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language, self.version)

        for i_text, text in enumerate(texts):
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if text[-1] not in self.splits:
                text += "。" if text_language != "en" else "."
            self.logger.info("{}: {}".format(self.i18n("实际输入的目标文本(每句)"), text))
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language, self.version)
            self.logger.info("{}: {}".format(self.i18n("前端处理后的文本(每句)"), norm_text2))
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

            t2 = ttime()
            # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
            # print(cache.keys(),if_freeze)
            if i_text in self.cache and if_freeze:
                pred_semantic = self.cache[i_text]
            else:
                with torch.no_grad():
                    pred_semantic, idx = self.t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        None if ref_free else prompt,
                        bert,
                        # prompt_phone_len=ph_offset,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=self.hz * self.max_sec(),
                    )
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    self.cache[i_text] = pred_semantic
            t3 = ttime()
            ###v3不存在以下逻辑和inp_refs
            if self.model_version not in {"v3", "v4"}:
                refers = []
                if inp_refs:
                    for path in inp_refs:
                        try:
                            refer = self.get_spepc(self.hps, path.name).to(self.dtype).to(self.device)
                            refers.append(refer)
                        except Exception:
                            traceback.print_exc()
                if len(refers) == 0:
                    refers = [self.get_spepc(self.hps, ref_wav_path).to(self.dtype).to(self.device)]
                audio = self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refers, speed=speed)[0][
                    0
                ]  # .cpu().detach().numpy()
            else:
                refer = self.get_spepc(self.hps, ref_wav_path).to(self.device).to(self.dtype)
                phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
                # print(11111111, phoneme_ids0, phoneme_ids1)
                fea_ref, ge = self.vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                ref_audio, sr = torchaudio.load(ref_wav_path)
                ref_audio = ref_audio.to(self.device).float()
                if ref_audio.shape[0] == 2:
                    ref_audio = ref_audio.mean(0).unsqueeze(0)
                tgt_sr = 24000 if self.model_version == "v3" else 32000
                if sr != tgt_sr:
                    ref_audio = self.resample(ref_audio, sr, tgt_sr)
                # print("ref_audio",ref_audio.abs().mean())
                mel2 = self.mel_fn(ref_audio) if self.model_version == "v3" else self.mel_fn_v4(ref_audio)
                mel2 = self.norm_spec(mel2)
                T_min = min(mel2.shape[2], fea_ref.shape[2])
                mel2 = mel2[:, :, :T_min]
                fea_ref = fea_ref[:, :, :T_min]
                Tref = 468 if self.model_version == "v3" else 500
                Tchunk = 934 if self.model_version == "v3" else 1000
                if T_min > Tref:
                    mel2 = mel2[:, :, -Tref:]
                    fea_ref = fea_ref[:, :, -Tref:]
                    T_min = Tref
                chunk_len = Tchunk - T_min
                mel2 = mel2.to(self.dtype)
                fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
                cfm_resss = []
                idx = 0
                while 1:
                    fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                    if fea_todo_chunk.shape[-1] == 0:
                        break
                    idx += chunk_len
                    fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                    cfm_res = self.vq_model.cfm.inference(
                        fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
                    )
                    cfm_res = cfm_res[:, :, mel2.shape[2] :]
                    mel2 = cfm_res[:, :, -T_min:]
                    fea_ref = fea_todo_chunk[:, :, -T_min:]
                    cfm_resss.append(cfm_res)
                cfm_res = torch.cat(cfm_resss, 2)
                cfm_res = self.denorm_spec(cfm_res)
                if self.model_version == "v3" and self.bigvgan_model is None:
                    self.bigvgan_model = self.init_bigvgan()
                else:  # v4
                    if self.hifigan_model is None:
                        self.hifigan_model = self.init_hifigan()
                vocoder_model = self.bigvgan_model if self.model_version == "v3" else self.hifigan_model
                with torch.inference_mode():
                    wav_gen = vocoder_model(cfm_res)
                    audio = wav_gen[0][0]  # .cpu().detach().numpy()
            max_audio = torch.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1:
                audio = audio / max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav_torch)  # zero_wav
            t4 = ttime()
            t.extend([t2 - t1, t3 - t2, t4 - t3])
            t1 = ttime()
        self.logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))
        audio_opt = torch.cat(audio_opt, 0)  # np.concatenate
        if self.model_version in {"v1", "v2"}:
            opt_sr = 32000
        elif self.model_version == "v3":
            opt_sr = 24000
        else:
            opt_sr = 48000  # v4
        if if_sr and opt_sr == 24000:
            self.logger.info(self.i18n("音频超分中"))
            audio_opt, opt_sr = self.audio_sr(audio_opt.unsqueeze(0), opt_sr)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
        else:
            audio_opt = audio_opt.cpu().detach().numpy()
        yield opt_sr, (audio_opt * 32767).astype(np.int16)

    def audio_sr(self, audio, sr):
        if self.sr_model is None:
            try:
                self.sr_model = AP_BWE(self.device, DictToAttrRecursive)
            except FileNotFoundError:
                logging.warning(self.i18n("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好"))
                return audio.cpu().detach().numpy(), sr
        return self.sr_model(audio, sr)

    def synthesize(
        self,
        GPT_model_path: str,
        SoVITS_model_path: str,
        ref_audio_path: str,
        ref_text_path: str,
        ref_language: str,
        target_text_path: str,
        target_language: str,
        output_path: Optional[str] = None,
    ):
        # Read reference text
        with open(ref_text_path, "r", encoding="utf-8") as file:
            ref_text = file.read()

        # Read target text
        with open(target_text_path, "r", encoding="utf-8") as file:
            target_text = file.read()

        if output_path is None:
            output_path = os.path.join(os.path.dirname(target_text_path), os.path.basename(target_text_path).split(".")[0] + ".wav")
            self.logger.info(output_path)

        # Change model weights
        self.config, self.t2s_model = self.change_gpt_weights(gpt_path=GPT_model_path)
        self.hps, self.vq_model = self.change_sovits_weights(sovits_path=SoVITS_model_path)

        # Synthesize audio
        synthesis_result = self.get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language=self.i18n(ref_language),
            text=target_text,
            text_language=self.i18n(target_language),
            top_p=1,
            temperature=1,
        )

        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = output_path
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            self.logger.info(f"Audio saved to {output_wav_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument(
        "--gpt_model",
        required=False,
        help="Path to the GPT model file",
        default="gptsovits/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    )
    parser.add_argument(
        "--sovits_model",
        required=False,
        help="Path to the SoVITS model file",
        default="gptsovits/pretrained_models/s2G488k.pth",
    )
    parser.add_argument("--ref_audio", required=True, help="Path to the reference audio file")
    parser.add_argument("--ref_text", required=True, help="Path to the reference text file")
    parser.add_argument(
        "--ref_language",
        required=False,
        choices=["中文", "英文", "日文"],
        help="Language of the reference audio",
        default="中文",
    )
    parser.add_argument("--target_text", required=True, help="Path to the target text file")
    parser.add_argument(
        "--target_language",
        required=False,
        choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"],
        help="Language of the target text",
        default="中文",
    )
    parser.add_argument("--output_path", required=False, help="Path to the output wav file", default=None)

    args = parser.parse_args()

    worker = Worker(gpt_path=args.gpt_model, sovits_path=args.sovits_model)
    worker.synthesize(
        args.gpt_model,
        args.sovits_model,
        args.ref_audio,
        args.ref_text,
        args.ref_language,
        args.target_text,
        args.target_language,
        args.output_path,
    )


if __name__ == "__main__":
    main()
