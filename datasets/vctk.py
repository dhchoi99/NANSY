import os

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

        wav_22k = utils.audio.load_wav(wav_path_22k, sr=22050)
        wav_22k = torch.from_numpy(wav_22k)
        wav_16k = utils.audio.load_wav(wav_path_16k, sr=16000)
        wav_16k = torch.from_numpy(wav_16k)

        return_data['audio'] = wav_22k
        return_data['audio_f'] = f(wav_16k, sr=16000)
        return_data['audio_g'] = g(wav_16k, sr=16000)

        text = data['text']
        return_data['text'] = text

        return return_data


if __name__ == '__main__':
    from omegaconf import OmegaConf

    conf = OmegaConf.load('configs/datasets/vctk.yaml')
    conf.mode = 'train'
    d = VCTKDataset(conf)
    print(len(d))
    data = d[0]
    print(data)
