import importlib

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Trainer(pl.LightningModule):
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.conf = conf

    def train_dataloader(self):
        conf_dataset = self.conf.datasets['train']
        module, cls = conf_dataset['class'].rsplit('.', 1)
        D = getattr(importlib.import_module(module, package=None), cls)
        train_dataset = D(conf_dataset)
        train_dataloader = DataLoader(train_dataset, shuffle=conf_dataset['shuffle'],
                                      batch_size=conf_dataset['batch_size'],
                                      num_workers=conf_dataset['num_workers'], drop_last=True, )

        return train_dataloader

    def val_dataloader(self):
        conf_dataset = self.conf.datasets['eval']
        module, cls = conf_dataset['class'].rsplit('.', 1)
        D = getattr(importlib.import_module(module, package=None), cls)
        eval_dataset = D(conf_dataset)
        eval_dataloader = DataLoader(eval_dataset, shuffle=conf_dataset['shuffle'],
                                     batch_size=conf_dataset['batch_size'],
                                     num_workers=conf_dataset['num_workers'], drop_last=True, )

        return eval_dataloader

    def build_models(self):
        networks = {}
        for key, conf_model in self.args.models.items():
            module, cls = conf_model['class'].rsplit(".", 1)
            M = getattr(importlib.import_module(module, package=None), cls)
            m = M(conf_model)

            networks[key] = m
        return networks

    def build_losses(self):
        losses_dict = {}
        losses_dict['L1'] = torch.nn.L1Loss()

        return losses_dict

    def configure_optimizers(self):
        optims = {}
        for key, conf_model in self.conf.models.items():
            if conf_model['optim'] is not None:
                conf_optim = conf_model['optim']
                module, cls = conf_optim['class'].rsplit(".", 1)
                O = getattr(importlib.import_module(module, package=None), cls)
                o = O([p for p in self.networks[key].parameters() if p.requires_grad],
                      lr=conf_optim.lr, betas=conf_optim.betas)

                optims[key] = o

        optim_module_keys = optims.keys()

        count = 0
        optim_list = []

        for _key in self.module_keys:
            if _key in optim_module_keys:
                optim_list.append(optims[_key])
                self.opt_tag[_key] = count
                count += 1

        return optim_list

    @property
    def automatic_optimization(self):
        return False
