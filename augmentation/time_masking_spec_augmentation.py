# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: time_masking_spec_augmentation.py

import random

import torch

from . import BaseAugmentation
import torchaudio.transforms as T


class TimeMaskingSpecAugmentation(BaseAugmentation):
    def _apply_augumentation(self, spectrogram):
        mask_parameter = random.choice([10, 20, 30, 40, 50, 60])
        time_masking = T.TimeMasking(time_mask_param=mask_parameter)
        return time_masking(spectrogram, mask_value=torch.min(spectrogram).item())
