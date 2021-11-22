import os

import torch

from datasets.custom import CustomDataset


class CSS10Dataset(CustomDataset):
    def __init__(self, conf):
        super(CSS10Dataset, self).__init__(conf)

        self.data = self.read_metadata()

    def read_metadata(self):
        path_metadata = self.conf.path.metadata
        with open(path_metadata, 'r') as f:
            file_lists = f.readlines()

        data_list = []
        for line in file_lists:
            wav_path, original_script, normalized_script, audio_duration = line.strip().split('|')
            wav_path_22k = os.path.join(self.conf.path.root, wav_path)
            data = {
                'wav_path_22k': wav_path_22k,
                'wav_path_16k': None,  # TODO
                'text': normalized_script,
            }
            data_list.append(data)

        # TODO split train/test set
        train_val_split_idx = int(len(data_list) * 0.9)
        mode = self.conf.mode
        if mode == 'train':
            data_list = data_list[:train_val_split_idx]
        elif mode == 'eval':
            data_list = data_list[train_val_split_idx:]
        elif mode == 'all':
            pass
        else:
            raise NotImplementedError

        return data_list


if __name__ == '__main__':
    from omegaconf import OmegaConf

    for file in os.listdir('configs/datasets'):
        if file.strip().endswith('yaml') and file.startswith('css10'):
            print(file)
            conf_c = OmegaConf.load(f'configs/datasets/{file}')
            conf_c.mode = 'train'
            d_c = CSS10Dataset(conf_c)
            print(len(d_c))
            data = d_c[0]

            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(key, value.shape)
                else:
                    print(key, type(value))
