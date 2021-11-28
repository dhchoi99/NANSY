import os
import random

import numpy as np
import librosa
import torch
import torchaudio.functional as AF

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
        self.mel_len = 128
        self.window_size = self.mel_len * self.conf.audio.hop_size - 1  # 32767, to make time-scale of mel to 128
        self.time_size = self.window_size / 22050  # 1.48810 (sec)
        self.window_size_16k = int(self.time_size * 16000)  # 23776

        self.yin_time = (self.window_size + self.w_len) / 22050.
        self.yin_window_size = self.window_size + self.w_len

        self.praat_voice_time = 0.2

        self.mel_safety_index = 40  # index to give segment enough voice values

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

    def load_mel(self, wav_path, sr=None):
        mel_path = wav_path + '.dhc.mel'
        try:
            mel = torch.load(mel_path, map_location='cpu')
        except Exception as e:
            print(e)
            wav_numpy, _ = self.load_wav(wav_path, sr)
            mel = torch.from_numpy(
                utils.audio.mel_from_audio(self.conf.audio, wav_numpy)
            ).float().permute((1, 0))
            torch.save(mel, mel_path)

        return mel

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
    def crop_audio(x: torch.Tensor, start: int, end: int, padding_value=0.):
        # x.shape: (..., T)
        if start < 0:
            if end > x.shape[-1]:
                y = x
                y = CustomDataset.pad_audio(y, -start, padding_value, pad_at='start')
                y = CustomDataset.pad_audio(y, end - x.shape[-1], padding_value, pad_at='end')
            else:
                y = x[..., :end]
                y = CustomDataset.pad_audio(y, -start, padding_value, pad_at='start')
        elif end > x.shape[-1]:
            y = x[..., start:]
            y = CustomDataset.pad_audio(y, end - x.shape[-1], padding_value, pad_at='end')
        else:
            y = x[..., start:end]
        assert y.shape[-1] == end - start, f'{x.shape}, {start}, {end}, {y.shape}'
        return y

    def getitem(self, idx):
        data = self.data[idx]

        return_data = {}
        # return_data.update(data)
        return_data['wav_path_22k'] = data['wav_path_22k']
        return_data['text'] = data['text']

        wav_22k_path = data['wav_path_22k']
        wav_16k_path = data['wav_path_16k']

        wav_22k_numpy, wav_22k_torch = self.load_wav(wav_22k_path, 22050)
        if wav_16k_path is not None and os.path.isfile(wav_16k_path):
            wav_16k_numpy, wav_16k_torch = self.load_wav(wav_16k_path, 16000)
        else:
            wav_16k_torch = AF.resample(wav_22k_torch, 22050, 16000)
            wav_16k_numpy = wav_16k_torch.numpy()

        mel_22k = self.load_mel(wav_22k_path, sr=22050)

        mel_start = random.randint(0, mel_22k.shape[-1] - self.mel_safety_index)
        mel_end = mel_start + self.mel_len
        gt_mel_22k = self.crop_audio(mel_22k, mel_start, mel_end, padding_value=-4)
        return_data['gt_mel_22k'] = gt_mel_22k

        t_start = mel_start * self.conf.audio.hop_size / 22050.
        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)
        w_end_22k = w_start_22k + self.window_size
        w_end_22k_yin = w_start_22k + self.yin_window_size
        w_end_16k = w_start_16k + self.window_size_16k
        assert w_start_22k <= wav_22k_torch.shape[-1], '22k_1'
        assert w_start_16k <= wav_16k_torch.shape[-1], '16k_1'

        wav_22k = self.crop_audio(wav_22k_torch, w_start_22k, w_end_22k)
        wav_16k = self.crop_audio(wav_16k_torch, w_start_16k, w_end_16k)
        wav_22k_yin = self.crop_audio(wav_22k_torch, w_start_22k, w_end_22k_yin)

        mel_start_negative = mel_start  # TODO for ablation: if diff(t_start, t_negative) > threshold
        threshold = 10  # at least 0.12 sec difference
        while abs(mel_start_negative - mel_start) < threshold:
            mel_start_negative = random.randint(0, mel_22k.shape[-1] - self.mel_safety_index)

        t_negative = mel_start_negative * self.conf.audio.hop_size / 22050.
        w_start_16k_negative = int(t_negative * 16000)
        w_end_16k_negative = w_start_16k_negative + self.window_size_16k
        assert w_start_16k_negative <= wav_16k_torch.shape[-1], "16k_nega!!!!!!!!!!!!!!"

        wav_16k_negative = self.crop_audio(wav_16k_torch, w_start_16k_negative, w_end_16k_negative)

        return_data['gt_audio_f'] = f(wav_16k, sr=16000)
        return_data['gt_audio_16k'] = wav_16k

        return_data['gt_audio_16k_negative'] = wav_16k_negative

        return_data['gt_audio_22k'] = wav_22k
        return_data['gt_audio_g'] = g(wav_22k_yin, sr=22050)

        return_data['t'] = t_start

        return return_data
