import importlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Trainer(pl.LightningModule):
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.conf = conf

        self.networks = torch.nn.ModuleDict(self.build_models())

        self.opt_tag = {key: None for key in self.networks.keys()}

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
        for key, conf_model in self.conf.models.items():
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
                      **conf_optim.kwargs)

                optims[key] = o

        optim_module_keys = optims.keys()

        count = 0
        optim_list = []

        for _key in self.networks.keys():
            if _key in optim_module_keys:
                optim_list.append(optims[_key])
                self.opt_tag[_key] = count
                count += 1

        return optim_list

    @property
    def automatic_optimization(self):
        return False

    def training_step(self, batch, batch_idx):
        loss = {}
        logs = {}

        lps = self.networks['Analysis'].linguistic(batch['audio_f'])
        s = self.networks['Analysis'].speaker(batch['audio_16k'])
        e = self.networks['Analysis'].energy(batch['mel_22k'])
        ps = self.networks['Analysis'].pitch.yingram_batch(batch['audio_g']).to(lps.device)
        ps = ps[:, 19:69]

        result = self.networks['Synthesis'](lps, s, e, ps)

        loss_mel = F.l1_loss(result['gen_mel'], batch['mel_22k'].permute((0, 2, 1)))

        opts = self.optimizers()
        opts[self.opt_tag['Analysis']].zero_grad()
        opts[self.opt_tag['Synthesis']].zero_grad()

        loss_mel.backward()

        opts[self.opt_tag['Analysis']].step()
        opts[self.opt_tag['Synthesis']].step()

    def validation_step(self, batch, batch_idx):
        loss = {}
        logs = {}

        lps = self.networks['Analysis'].linguistic(batch['audio_f'])
        s = self.networks['Analysis'].speaker(batch['audio_16k'])
        e = self.networks['Analysis'].energy(batch['mel_22k'])
        ps = self.networks['Analysis'].pitch.yingram_batch(batch['audio_g']).to(lps.device)
        ps = ps[:, 19:69]

        result = self.networks['Synthesis'](lps, s, e, ps)

        loss_mel = F.l1_loss(result['gen_mel'], batch['mel_22k'].permute((0, 2, 1)))
