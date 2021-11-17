import importlib

from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from utils.conf import set_conf


class BaseDataset(Dataset):
    def __init__(self, conf):
        super(BaseDataset, self).__init__()
        self.conf = set_conf(conf)

    # region init

    # endregion

    # region getitem

    def getitem(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        result = None
        while result is None:
            try:
                result = self.getitem(idx)
                return result
            except Exception as e:
                raise e
                # print(f'error {e} on idx {idx}')
                # idx = random.randint(0, len(self))

    # endregion

    def __len__(self):
        raise NotImplementedError


class MultiDataset(BaseDataset):
    def __init__(self, conf):
        super(MultiDataset, self).__init__(conf)

        self.datasets, self.idx2item = self.build_datasets(self.conf.datasets)

    def build_datasets(self, conf):
        idx2item = []
        datasets = {}
        for path_conf in conf:
            conf_dataset = OmegaConf.load(path_conf)
            conf_dataset.mode = self.conf.mode  # TODO
            module, cls = conf_dataset['class'].rsplit(".", 1)
            D = getattr(importlib.import_module(module, package=None), cls)
            # key = conf_dataset.id
            key = cls
            d = D(conf_dataset)
            datasets[key] = d
            idx2item += [(key, idx) for idx in range(len(d))]

        return datasets, idx2item

    def __len__(self):
        return len(self.idx2item)

    def getitem(self, idx):
        key, idx = self.idx2item[idx]
        return self.datasets[key][idx]
