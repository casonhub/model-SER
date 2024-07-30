# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: freq_masking_spec_augmentation.py

import random

import torch

from . import BaseAugmentation
import torchaudio.transforms as T

class FreqMaskingSpecAugmentation(BaseAugmentation):
    def _apply_augumentation(self, spectrogram):
        mask_parameter = random.choice([6, 8, 10, 12, 14, 16])
        freq_masking = T.FrequencyMasking(freq_mask_param=mask_parameter)
        return freq_masking(spectrogram, mask_value=torch.min(spectrogram).item())
