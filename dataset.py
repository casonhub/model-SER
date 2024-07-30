# -*- encoding: utf-8 -*-
# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 31/1/2023
# @File: base_augmentation.py

import torch
import torchaudio

class SpeechRecordingsSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, labels, target_signal_sr, n_mels, n_frames, normalize=False, mean=None, std=None, signal_augumentation=None, spectrogram_augumentation=None):
        assert (normalize and mean is not None and std is not None) or not normalize
        self.filepaths = filepaths
        self.labels = labels
        self.target_signal_sr = target_signal_sr
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.signal_augumentation = signal_augumentation
        self.spectrogram_augumentation = spectrogram_augumentation

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        signal, sample_rate = torchaudio.load(self.filepaths[idx])

        if (signal.shape[0] > 1):
            signal = signal[0, :]

        # adjust signal sample rate
        signal = self._adjust_signal_sr(signal, sample_rate)
        if self.signal_augumentation:
            signal = self.signal_augumentation(signal)

        # kaldi mel spectrogram
        spectrogram = torchaudio.compliance.kaldi.fbank(signal.unsqueeze(0), htk_compat=True, sample_frequency=self.target_signal_sr, use_energy=False, window_type='hanning', num_mel_bins=self.n_mels, dither=0.0, frame_shift=10).T

        if self.normalize:
            spectrogram = (spectrogram - self.mean) / (self.std * 2)
        # because of spectrogram augumentation -> needs (1, self.n_mels, n_frames)
        # DONT FORGET, AFTER THIS TRANSFORMATION WILL HAVE SPECTROGRAM SHAPE: (1, self.n_mels, n_frames), 
        # n_frames is not equal yet to self.n_frames, adjusted in next step
        spectrogram = spectrogram.unsqueeze(0)
        # pad spectrogram to given self.n_frams
        if self.spectrogram_augumentation:
            spectrogram = self.spectrogram_augumentation(spectrogram)

        if self.n_frames:
            if spectrogram.shape[-1] < self.n_frames:
                min_value = torch.min(spectrogram)
                pad_size = self.n_frames - spectrogram.size()[-1]
                spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_size), value=min_value)
            else:
                # ONE MORE TIME -> spectrogram.shape has three dimensions (1, ..., ...)
                spectrogram = spectrogram[:, :, :self.n_frames]


        return spectrogram.squeeze(0).T, self.labels[idx]

    def _adjust_signal_sr(self, signal, sr):
        if self.target_signal_sr == sr:
            return signal.squeeze(0)
        resample = torchaudio.transforms.Resample(sr, self.target_signal_sr)
        return resample(signal).squeeze(0)