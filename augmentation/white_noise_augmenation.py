# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: white_noise_augmentation.py

import random

import torch

from . import BaseAugmentation


class RandomWhiteNoiseAugmentation(BaseAugmentation):
    def _apply_augumentation(self, recording):
        noise_factor = random.choice([0.1, 0.15, 0.2])

        noise = torch.normal(0, recording.std(), size=(1,recording.shape[0])).squeeze(0)
        return recording + noise * noise_factor
