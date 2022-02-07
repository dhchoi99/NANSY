import os
import random

import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.functional as AF

from datasets.base import BaseDataset
from datasets.functional import f, g, get_pitch_median
import utils.mel


class CustomDataset(BaseDataset):
    def __init__(self, conf):
        super(CustomDataset, self).__init__(conf)

        self.configure_args()

        self.data = []
        # data: list
        # data[i]: dict must have:
        # 'wav_path_22k': str = path_to_22k_wav_file
        # 'wav_path_16k': str = path_to_16k_wav_file
        # 'speaker_id': str = speaker_id

    # region init

    def configure_args(self):
        self.w_len = 2048  # for yingram
        self.mel_window = 128  # window size in mel
        self.audio_window_22k = self.mel_window * self.conf.audio.hop_size  # 32768, window size in raw audio
        self.segment_duration = self.audio_window_22k / 22050  # 1.486 (sec)
        self.audio_window_16k = int(self.segment_duration * 16000)  # 23778

        self.yin_window_22k = self.audio_window_22k + self.w_len  # 34816
        self.yin_segment_duration = self.yin_window_22k / 22050.  # 1.579

        zero_audio = torch.zeros(self.yin_window_22k).float()
        zero_mel = self.load_mel_from_audio(zero_audio, self.conf.audio)
        self.mel_padding_value = torch.min(zero_mel).data

        self.minimum_audio_length = self.yin_window_22k
        self.minimum_mel_length = zero_mel.shape[-1]

    # endregion

    def __len__(self):
        return len(self.data)

    # region getitem

    @staticmethod
    def load_wav(path: str, sr: int = None):
        r"""given str path, loads wav (supports raw audio, .npy, .pt dataform)

        params:
            path: path to audio data file
            sr: sampling_rate, used only when path given to raw audio

        returns:
            wav_numpy: numpy.ndarray of shape (n, )
            wav_torch: torch.Tensor of shape (n,)
        """
        if path.endswith('.pt'):
            wav_torch = torch.load(path)
            wav_numpy = wav_torch.numpy()
        elif path.endswith('.npy'):
            wav_numpy = np.load(path)
            wav_torch = torch.from_numpy(wav_numpy).float()
        else:
            assert sr is not None
            try:
                wav_numpy, sr = librosa.core.load(path, sr=sr)
                wav_torch = torch.from_numpy(wav_numpy).float()
            except:
                raise ValueError(f"could not load audio file from path :{path}")
        return wav_numpy, wav_torch

    def load_mel_from_audio(self, wav_torch: torch.Tensor, conf_audio: dict = None):
        """calculates mel from torch.Tensor audio

        Args:
            wav_torch: torch.Tensor of shape (n,) or (B, n)
            conf_audio

        Returns:
            mel: torch.Tensor of shape (C x T) or (B x C X T)

        """
        if conf_audio is None:
            conf_audio = self.conf.audio

        if wav_torch.ndim == 1:
            wav = wav_torch.unsqueeze(0)  # 1 x n
        elif wav_torch.ndim == 2:
            wav = wav_torch
        else:
            raise NotImplementedError

        mel = utils.mel.mel_spectrogram(
            wav,
            conf_audio['n_fft'],
            conf_audio['num_mels'],
            conf_audio['sample_rate'],
            conf_audio['hop_size'],
            conf_audio['win_size'],
            conf_audio['fmin'],
            conf_audio['fmax']
        )  # B x C x T

        if wav_torch.ndim == 1:
            mel = mel[0]  # 1 x C x T -> C x T
        return mel

    def load_mel(self, path_audio: str, sr: int = None, wav_torch=None) -> torch.Tensor:
        r"""hifi-gan style mel loading

        params:
            path_audio: path to audio data file
            sr: sampling rate

        returns:
            mel: torch.Tensor of shape (C x T)
        """
        mel_path = path_audio + '.mel'
        if False:
            pass
        # if os.path.exists(mel_path):
        #     mel = torch.load(mel_path, map_location='cpu')
        else:
            if wav_torch is None:
                _, wav_torch = self.load_wav(path_audio, sr=sr)
            mel = self.load_mel_from_audio(wav_torch, self.conf.audio)
            # torch.save(mel, mel_path)
        return mel

    @staticmethod
    def pad_audio(x: torch.Tensor, length: int, value: float = 0., pad_at: str = 'end') -> torch.Tensor:
        r"""pads value to audio data, at last dimension

        params:
            x: torch.Tensor of shape (..., T)
            length: int, length to pad
            value: float, value to pad
            pad_at: str, 'start' or 'end'

        returns:
            y: padded torch.Tensor of shape (..., T+length)
        """
        # x: (..., T)
        pad_at = pad_at.strip().lower()
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
    def crop_audio(x: torch.Tensor, start: int, end: int, padding_value: float = 0.) -> torch.Tensor:
        r"""crop audio data at last dimension from start to end, automatically pad with padding_value

        params:
            x: torch.Tensor of shape (..., T)
            start: int, position to crop
            end: int, position to crop
            padding_value: float, value for padding when needed

        returns:
            y: torch.Tensor of shape (..., end-start)
        """
        if start < 0:
            if end < 0:
                y = torch.ones(size=(*x.shape[:-1], end - start), dtype=torch.float, device=x.device) * padding_value
            elif end > x.shape[-1]:
                y = x
                y = CustomDataset.pad_audio(y, -start, padding_value, pad_at='start')
                y = CustomDataset.pad_audio(y, end - x.shape[-1], padding_value, pad_at='end')
            else:
                y = x[..., :end]
                y = CustomDataset.pad_audio(y, -start, padding_value, pad_at='start')
        elif end > x.shape[-1]:
            if start > x.shape[-1]:
                y = torch.ones(size=(*x.shape[:-1], end - start), dtype=torch.float, device=x.device) * padding_value
            else:
                y = x[..., start:]
                y = CustomDataset.pad_audio(y, end - x.shape[-1], padding_value, pad_at='end')
        else:
            y = x[..., start:end]
        assert y.shape[-1] == end - start, f'{x.shape}, {start}, {end}, {y.shape}'
        return y

    def get_wav_22k(self, wav_22k_path: str):
        wav_22k_numpy, wav_22k_torch = self.load_wav(wav_22k_path, 22050)
        return wav_22k_numpy, wav_22k_torch

    def get_wav_16k(self, wav_16k_path: str, wav_22k_path: str = None, wav_22k_torch: torch.Tensor = None):
        r"""loads 16k audio

        if 16k audio file exists, load audio
        else, resample from 22k audio
        """

        if wav_22k_torch is not None:
            # wav_16k_torch = AF.resample(wav_22k_torch, 22050, 16000)
            wav_16k_torch = torchaudio.transforms.Resample(22050, 16000).forward(wav_22k_torch)
            wav_16k_numpy = wav_16k_torch.numpy()
        elif wav_16k_path is not None and os.path.isfile(wav_16k_path):
            wav_16k_numpy, wav_16k_torch = self.load_wav(wav_16k_path, 16000)
        else:
            raise NotImplementedError
        return wav_16k_numpy, wav_16k_torch

    def get_time_idxs(self, mel_start: int):
        r"""calculates time-related idxs needed for getitem

        params:
            mel_start: idx where splitted mel starts

        returns:
            mel_start: int, start index of mel
            mel_end: int, end index of mel
            t_start: float, start time (sec)
            w_start_16k: int, start index of 16k audio
            w_start_22k: int, start index of 22k audio
            w_end_16k: int, end index of 16k audio
            w_end_22k: int, end index of 22k audio
            w_end_22k_yin: int, end index of 22k audio for yin computation
        """
        mel_end = mel_start + self.mel_window

        t_start = mel_start * self.conf.audio.hop_size / 22050.
        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)
        w_end_22k = w_start_22k + self.audio_window_22k
        w_end_22k_yin = w_start_22k + self.yin_window_22k
        w_end_16k = w_start_16k + self.audio_window_16k

        return mel_start, mel_end, t_start, w_start_16k, w_start_22k, w_end_16k, w_end_22k, w_end_22k_yin

    def get_pos_sample(self, data: dict):
        r"""loads positive sample from data

        params:
            data: dict, item from self.data

        returns:
            return_data: dict, positive sample related data
        """
        return_data = {}
        return_data['wav_path_22k'] = data['wav_path_22k']
        return_data['text'] = data['text']

        wav_22k_numpy, wav_22k_torch = self.get_wav_22k(data['wav_path_22k'])
        if wav_22k_torch.shape[-1] < self.minimum_audio_length:
            wav_22k_torch = torch.nn.functional.pad(
                wav_22k_torch, (0, self.minimum_audio_length - wav_22k_torch.shape[-1]), mode='constant', value=0.0)

        assert wav_22k_torch.shape[-1] >= self.minimum_audio_length, f'{wav_22k_numpy.shape}'

        _, pitch_median = get_pitch_median(wav_22k_numpy, sr=22050)
        return_data['pitch_median_pos'] = pitch_median
        _, wav_16k_torch = self.get_wav_16k(data['wav_path_16k'], data['wav_path_22k'], wav_22k_torch)
        mel_22k = self.load_mel(data['wav_path_22k'], sr=22050, wav_torch=wav_22k_torch)

        # assert mel_22k.shape[-1] >= self.minimum_mel_length, f'{mel_22k.shape[-1]}, {self.minimum_mel_length}'
        mel_start = random.randint(0, mel_22k.shape[-1] - self.minimum_mel_length)
        pos_time_idxs = self.get_time_idxs(mel_start)

        assert mel_22k.shape[-1] >= pos_time_idxs[1]
        mel_22k = self.crop_audio(mel_22k, pos_time_idxs[0], pos_time_idxs[1])
        return_data['gt_mel_22k'] = mel_22k

        assert pos_time_idxs[5] <= wav_16k_torch.shape[-1], '16k_1'
        wav_16k = self.crop_audio(wav_16k_torch, pos_time_idxs[3], pos_time_idxs[5])
        return_data['gt_audio_16k'] = wav_16k
        wav_16k_torch_f = f(wav_16k_torch, sr=16000)
        return_data['gt_audio_16k_f'] = self.crop_audio(wav_16k_torch_f, pos_time_idxs[3], pos_time_idxs[5])

        assert pos_time_idxs[7] <= wav_22k_torch.shape[-1], '22k_1'
        wav_22k = self.crop_audio(wav_22k_torch, pos_time_idxs[4], pos_time_idxs[6])
        return_data['gt_audio_22k'] = wav_22k
        wav_22k_torch_g = g(wav_22k_torch, sr=22050)
        return_data['gt_audio_22k_g'] = self.crop_audio(wav_22k_torch_g, pos_time_idxs[4], pos_time_idxs[7])

        return return_data

    def get_neg_sample(self, data: dict):
        return_data = {}

        wav_22k_numpy, wav_22k_torch = self.get_wav_22k(data['wav_path_22k'])
        if wav_22k_torch.shape[-1] < self.minimum_audio_length:
            wav_22k_torch = torch.nn.functional.pad(
                wav_22k_torch, (0, self.minimum_audio_length - wav_22k_torch.shape[-1]), mode='constant', value=0.0)

        _, pitch_median = get_pitch_median(wav_22k_numpy, sr=22050)
        return_data['pitch_median_neg'] = pitch_median
        _, wav_16k_torch = self.get_wav_16k(data['wav_path_16k'], data['wav_path_22k'], wav_22k_torch)
        mel_22k = self.load_mel(data['wav_path_22k'], sr=22050, wav_torch=wav_22k_torch)

        # assert mel_22k.shape[-1] >= self.minimum_mel_length
        mel_start = random.randint(0, mel_22k.shape[-1] - self.minimum_mel_length)
        negative_time_idxs = self.get_time_idxs(mel_start)

        assert negative_time_idxs[5] <= wav_16k_torch.shape[-1], "16k_nega"
        wav_16k_negative = self.crop_audio(wav_16k_torch, negative_time_idxs[3], negative_time_idxs[5])

        assert negative_time_idxs[6] <= wav_22k_torch.shape[-1], "22k_nega"
        wav_22k_negative = self.crop_audio(wav_22k_torch, negative_time_idxs[4], negative_time_idxs[6])

        return_data['gt_audio_16k_negative'] = wav_16k_negative
        return_data['gt_audio_22k_negative'] = wav_22k_negative
        return return_data

    def getitem(self, pos_idx: int):
        r"""

        params:
            pos_idx: int
        returns:
            return_data: dict
        """
        pos_data = self.data[pos_idx]

        # get negative sample idx: speaker_id should be different
        neg_idx = random.randint(0, len(self) - 1)
        neg_data = self.data[neg_idx]
        while neg_data['speaker_id'] == pos_data['speaker_id']:
            neg_idx = random.randint(0, len(self) - 1)
            neg_data = self.data[neg_idx]

        return_data = {}
        return_data_pos = self.get_pos_sample(pos_data)
        return_data_neg = self.get_neg_sample(neg_data)
        return_data.update(return_data_pos)
        return_data.update(return_data_neg)

        return return_data

    # endregion
