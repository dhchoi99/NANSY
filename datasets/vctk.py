import os

import torch

from datasets.custom import CustomDataset


class VCTKDataset(CustomDataset):
    def __init__(self, conf):
        super(VCTKDataset, self).__init__(conf)

        self.data = self.read_metadata()

    def read_metadata(self):
        mode = self.conf.mode

        path_timestamps = self.conf.path.timestamp

        timestamps = {}
        with open(path_timestamps, 'r') as f:
            timestamps_list = f.readlines()
        for line in timestamps_list:
            timestamp_data = line.strip().split(' ')
            if len(timestamp_data) == 3:
                file_id, t_start, t_end = timestamp_data
                timestamps[file_id] = (float(t_start), float(t_end))

        path_metadata = self.conf.path.configs[mode]

        with open(path_metadata, 'r') as f:
            file_lists = f.readlines()

        data_list = []

        for line in file_lists:
            wav_path, txt, speaker_id = line.split('|')
            file_id = os.path.split(wav_path)[-1].split('-22k')[0]
            wav_path_22k = os.path.join(self.conf.path.root, wav_path)
            wav_path_16k = wav_path_22k.replace('wav22', 'wav16')

            if file_id in timestamps.keys():
                data = {
                    'wav_path_22k': wav_path_22k,
                    'wav_path_16k': wav_path_16k,
                    'text': txt,
                    'timestamp': timestamps[file_id]
                }
                data_list.append(data)

        return data_list


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
