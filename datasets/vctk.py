import os
import random

import torch

from datasets.base import BaseDataset
from datasets.functional import f, g
import utils.audio


class VCTKDataset(BaseDataset):
    def __init__(self, conf):
        super(VCTKDataset, self).__init__(conf)

        self.conf = conf

        self.data = self.read_metadata()

    def read_metadata(self):
        mode = self.conf.mode
        path_metadata = self.conf.path.configs[mode]

        with open(path_metadata, 'r') as f:
            file_lists = f.readlines()

        data_list = []

        for line in file_lists:
            wav_path, txt, speaker_id = line.split('|')
            data = {
                'wav_path': wav_path,
                'text': txt
            }
            data_list.append(data)

        return data_list

    def __len__(self):
        return len(self.data)

    def getitem(self, idx):
        data = self.data[idx]

        return_data = {}

        wav_path_22k = os.path.join(self.conf.path.root, data['wav_path'])
        return_data['wav_path'] = wav_path_22k
        wav_path_16k = wav_path_22k.replace('wav22', 'wav16')

        wav_22k_numpy = utils.audio.load_wav(wav_path_22k, sr=22050)
        wav_22k_torch = torch.from_numpy(wav_22k_numpy)
        wav_16k_numpy = utils.audio.load_wav(wav_path_16k, sr=16000)
        wav_16k_torch = torch.from_numpy(wav_16k_numpy)

        w_len = 2048  # for yingram
        window_size = 128 * 256 - 1
        time_size = window_size / 22050  # 1.48810
        window_size_16k = int(time_size * 16000)
        assert len(wav_22k_torch) > window_size

        t_start = random.uniform(0, (len(wav_22k_torch) - window_size - w_len) / 22050.)
        t_end = t_start + time_size

        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)

        wav_22k = wav_22k_torch[w_start_22k:w_start_22k + window_size]
        wav_16k = wav_16k_torch[w_start_16k:w_start_16k + window_size_16k]
        wav_22k_yin = wav_22k_torch[w_start_22k:w_start_22k + window_size + w_len]

        text = data['text']
        return_data['text'] = text

        return_data['audio_16k'] = wav_16k.float()
        return_data['audio_22k'] = wav_22k.float()
        return_data['audio_f'] = f(wav_16k, sr=16000)[0].float()
        return_data['audio_g'] = g(wav_22k_yin, sr=16000)[0].float()
        # return_data['audio_yin'] = wav_22k_yin
        return_data['mel_22k'] = torch.from_numpy(utils.audio.mel_from_audio(self.conf.audio, wav_22k.numpy())).float()

        return return_data


if __name__ == '__main__':
    from omegaconf import OmegaConf

    conf = OmegaConf.load('configs/datasets/vctk.yaml')
    conf.mode = 'train'
    d = VCTKDataset(conf)
    print(len(d))
    data = d[0]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, type(value))
