# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 15/4/2023
# @File: speed_change_augmentation.py

import random

import torchaudio

from . import BaseAugmentation


class RandomSpeedChangeAugmentation(BaseAugmentation):
    def _apply_augumentation(self, recording):
        speed_factor = random.choice([0.85, 0.9, 1.1, 1.15])
        # print(f"SELECTED SPEED FACTOR: {speed_factor}")
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self._sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            recording.unsqueeze(0), self._sample_rate, sox_effects)
        # print(f"RETURNING RECORDING WITH SHAPE: {transformed_audio.shape}")
        return transformed_audio.squeeze(0)
