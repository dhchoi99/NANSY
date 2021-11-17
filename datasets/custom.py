import os
import random

import numpy as np
import librosa
import torch

from datasets.base import BaseDataset
from datasets.functional import f, g
import utils.audio


class CustomDataset(BaseDataset):
    def __init__(self, conf):
        super(CustomDataset, self).__init__(conf)

        self.configure_args()

        self.data = None
        # data: list
        # data[i]: dict should have:
        # 'timestamp': tuple = (t_start(default=0), t_end(default=inf))
        # 'wav_path_22k': str = path_to_22k_wav_file
        # 'wav_path_16k': str = path_to_16k_wav_file

    def configure_args(self):
        self.w_len = 2048  # for yingram
        self.window_size = 128 * 256 - 1  # 32767, to make time-scale of mel to 128
        self.time_size = self.window_size / 22050  # 1.48810 (sec)
        self.window_size_16k = int(self.time_size * 16000)  # 23776

        self.yin_window_size = self.window_size + self.w_len
        self.yin_window = (self.window_size + self.w_len) / 22050.

        self.praat_voice_time = 0.2

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_wav(path, sr=None):
        if path.endswith('.pt'):
            wav_torch = torch.load(path)
            wav_numpy = wav_torch.numpy()
        elif path.endswith('.npy'):
            wav_numpy = np.load(path)
            wav_torch = torch.from_numpy(wav_numpy).float()
        else:
            assert sr is not None
            wav_numpy, sr = librosa.core.load(path, sr=sr)
            wav_torch = torch.from_numpy(wav_numpy).float()
        return wav_numpy, wav_torch

    def getitem(self, idx):
        data = self.data[idx]

        return_data = {}

        path_wav_22k = data['wav_path_22k']
        path_wav_16k = data['wav_path_16k']

        wav_22k_numpy, wav_22k_torch = self.load_wav(path_wav_22k, 22050)
        wav_16k_numpy, wav_16k_torch = self.load_wav(path_wav_16k, 16000)

        t_audio_start = max(0.,
                            data['timestamp'][0] + self.praat_voice_time - self.time_size)

        t_audio_end = min((wav_22k_torch.shape[-1] - self.yin_window_size) / 22050.,
                          data['timestamp'][1] - self.praat_voice_time)

        t_start = random.uniform(t_audio_start, t_audio_end)
        # t_start = t_audio_end

        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)
        wav_22k = wav_22k_torch[w_start_22k:w_start_22k + self.window_size]
        wav_16k = wav_16k_torch[w_start_16k:w_start_16k + self.window_size_16k]

        assert (w_start_22k + self.yin_window_size) <= wav_22k_torch.shape[-1]
        wav_22k_yin = wav_22k_torch[w_start_22k:w_start_22k + self.yin_window_size]
        assert wav_22k_yin.shape[-1] == self.yin_window_size

        return_data['audio_f'] = f(wav_16k, sr=16000)[0].float()
        return_data['audio_16k'] = wav_16k

        t_negative = t_start
        while t_negative == t_start:
            t_negative = random.uniform(t_audio_start, t_audio_end)
        w_start_16k_negative = int(t_negative * 16000)
        wav_16k_negative = wav_16k_torch[w_start_16k_negative:w_start_16k_negative + self.window_size_16k]
        return_data['audio_16k_negative'] = wav_16k_negative

        return_data['audio_22k'] = wav_22k
        return_data['audio_g'] = g(wav_22k_yin, sr=22050)[0].float()

        return_data['mel_22k'] = torch.from_numpy(
            utils.audio.mel_from_audio(self.conf.audio, wav_22k.numpy())
        ).float().permute((1, 0))
        return_data['t'] = t_start

        return return_data
