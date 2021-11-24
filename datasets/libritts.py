import os

import torch

from datasets.custom import CustomDataset


class LibriTTSDataset(CustomDataset):
    def __init__(self, conf):
        super(LibriTTSDataset, self).__init__(conf)

        self.data = self.read_metadata()

    def read_metadata(self):
        mode = self.conf.mode

        path_metadata = self.conf.path.configs[mode]
        with open(path_metadata, 'r') as f:
            file_lists = f.readlines()

        data_list = []
        for line in file_lists:
            wav_path, txt, speaker_id = line.split('|')

            wav_path_22k = os.path.join(self.conf.path.root, wav_path)
            data = {
                'wav_path_22k': wav_path_22k,
                'wav_path_16k': None,  # TODO
                'text': txt,
            }
            data_list.append(data)

        return data_list


if __name__ == '__main__':
    from omegaconf import OmegaConf

    conf = OmegaConf.load('configs/datasets/libritts360.yaml')
    conf.mode = 'train'

    d = LibriTTSDataset(conf)
    print(len(d))
    data = d[0]

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(key, value.shape)
        else:
            print(key, type(value))
