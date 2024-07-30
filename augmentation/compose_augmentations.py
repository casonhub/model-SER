# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: compose_augmentations.py

class ComposeAugmentations:
    def __init__(self, transformations):
        self._t = transformations

    def __call__(self, recording):
        transformed = recording
        for t in self._t:
            transformed = t(transformed)
            # print(f"augumented recording shape: {transformed.shape}")

        return transformed
