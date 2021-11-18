import os
import random

import numpy as np
import librosa
import torch

from datasets.base import BaseDataset
from datasets.functional import f, g
import utils.audio
from utils.mel import mel_spectrogram


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
        self.mel_len = 128
        self.window_size = self.mel_len * self.conf.audio.hop_size - 1  # 32767, to make time-scale of mel to 128
        self.time_size = self.window_size / 22050  # 1.48810 (sec)
        self.window_size_16k = int(self.time_size * 16000)  # 23776

        self.yin_time = (self.window_size + self.w_len) / 22050.
        self.yin_window_size = self.window_size + self.w_len

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

    @staticmethod
    def pad_audio(x: torch.Tensor, length: int, value=0., pad_at='end'):
        # x: (..., T)
        if pad_at == 'end':
            y = torch.cat([
                x, torch.ones(*x.shape[:-1], length) * value
            ], dim=-1)
        elif pad_at == 'start':
            y = torch.cat([
                torch.ones(*x.shape[:-1], length) * value, x
            ], dim=-1)
        else:
            raise NotImplementedError
        return y

    @staticmethod
    def crop_audio(x: torch.Tensor, start: int, end: int, value=0.):
        # x.shape: (..., T)
        if start < 0:
            y = x[..., :end]
            y = CustomDataset.pad_audio(y, -start, value, pad_at='start')
        elif end > x.shape[-1]:
            y = x[..., start:]
            y = CustomDataset.pad_audio(y, end - x.shape[-1], value, pad_at='end')
        else:
            y = x[..., start:end]
        assert y.shape[-1] == end - start, f'{x.shape}, {start}, {end}, {y.shape}'
        return y

    def getitem(self, idx):
        data = self.data[idx]

        return_data = {}
        return_data.update(data)

        wav_22k_path = data['wav_path_22k']
        wav_16k_path = data['wav_path_16k']

        wav_22k_numpy, wav_22k_torch = self.load_wav(wav_22k_path, 22050)
        wav_16k_numpy, wav_16k_torch = self.load_wav(wav_16k_path, 16000)

        mel_22k_path = wav_22k_path + '.dhc.mel'
        if not os.path.exists(mel_22k_path):
            mel_22k = torch.from_numpy(
                utils.audio.mel_from_audio(self.conf.audio, wav_22k_numpy)
            ).float().permute((1, 0))
            torch.save(mel_22k, mel_22k_path)
        else:
            mel_22k = torch.load(mel_22k_path, map_location='cpu')

        t_min = max(data['timestamp'][0], 0)
        t_max = min(data['timestamp'][1], wav_22k_torch.shape[-1] / 22050.)
        t_start = random.uniform(t_min - self.time_size + self.praat_voice_time, t_max - 0.2)  # 0.2 for safety index

        # mel_start = random.randint(0, mel_22k.shape[-1] - 30)  # 30 for safety index
        mel_start = int(t_start * (22050 / self.conf.audio.hop_size))
        mel_end = mel_start + self.mel_len
        gt_mel_22k = self.crop_audio(mel_22k, mel_start, mel_end, value=int(torch.min(mel_22k)))
        return_data['gt_mel_22k'] = gt_mel_22k

        t_start = mel_start * self.conf.audio.hop_size / 22050.

        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)
        w_end_22k = w_start_22k + self.window_size
        w_end_22k_yin = w_start_22k + self.yin_window_size
        w_end_16k = w_start_16k + self.window_size_16k

        wav_22k = self.crop_audio(wav_22k_torch, w_start_22k, w_end_22k)
        wav_16k = self.crop_audio(wav_16k_torch, w_start_16k, w_end_16k)
        wav_22k_yin = self.crop_audio(wav_22k_torch, w_start_22k, w_end_22k_yin)

        # t_negative = t_start
        # while t_negative == t_start:  # TODO for ablation: if diff(t_start, t_negative) > threshold
        #     t_negative = random.uniform(t_min - self.time_size + self.praat_voice_time, t_max)
        mel_start_negative = mel_start
        while mel_start_negative == mel_start:
            mel_start_negative = random.randint(0, mel_22k.shape[-1])
        t_negative = mel_start_negative * self.conf.audio.hop_size / 22050.

        w_start_16k_negative = int(t_negative * 16000)
        w_end_16k_negative = w_start_16k_negative + self.window_size_16k
        wav_16k_negative = self.crop_audio(wav_16k_torch, w_start_16k_negative, w_end_16k_negative)

        return_data['gt_audio_f'] = f(wav_16k, sr=16000)[0].float()
        return_data['gt_audio_16k'] = wav_16k

        return_data['gt_audio_16k_negative'] = wav_16k_negative

        return_data['gt_audio_22k'] = wav_22k
        return_data['gt_audio_g'] = g(wav_22k_yin, sr=22050)[0].float()

        return_data['t'] = t_start

        return return_data