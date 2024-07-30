# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: rir_augmentation.py

import torch
import torchaudio

from . import BaseAugmentation


class RIRAugmentation(BaseAugmentation):
    def _apply_augumentation(self, recording):
        rir_path = "assets/sample_RIR_8kHz.wav"
        rir_raw, sample_rate = torchaudio.load(rir_path)

        rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
        rir_norm = rir / torch.norm(rir, p=2)
        rir_linalg_norm = rir / torch.linalg.norm(rir, ord=2)

        rir = torch.flip(rir, [1])
        torch.eq(rir_norm.squeeze(0), rir_linalg_norm.squeeze(0))

        augumented_recording = torch.nn.functional.pad(recording, (rir.shape[1] - 1, 0))
        return torch.nn.functional.conv1d(augumented_recording[None, ...], rir[None, ...])[0]
