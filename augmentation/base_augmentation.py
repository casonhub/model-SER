# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: base_augmentation.py

import torch


class BaseAugmentation(torch.nn.Module):
    def __init__(self, p, sample_rate):
        super().__init__()
        self._p = p
        self._sample_rate = sample_rate

    def forward(self, recording):
        should_apply = torch.bernoulli(torch.tensor(self._p))
        # print(f"Is transformation applied?: {should_apply}")
        if should_apply:
            a = self._apply_augumentation(recording)
            # print(f"PARENT CLASS FORWARD METHOD, RETURNING DATA WITH SHAPE: {a}")
            return a

        return recording

    def _apply_augumentation(self, _):
        raise NotImplementedError
