import importlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

MATPLOTLIB_FLAG = False


class Trainer(pl.LightningModule):
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.conf = conf

        self.networks = torch.nn.ModuleDict(self.build_models())

        self.opt_tag = {key: None for key in self.networks.keys()}

        self.losses = self.build_losses()

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
        losses_dict['BCE'] = torch.nn.BCEWithLogitsLoss()

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

    def common_step(self, batch, batch_idx):
        loss = {}
        logs = {}
        logs.update(batch)

        logs['lps'] = self.networks['Analysis'].linguistic(batch['gt_audio_f'])
        logs['s_pos'] = self.networks['Analysis'].speaker(batch['gt_audio_16k'])
        logs['s_neg'] = self.networks['Analysis'].speaker(batch['gt_audio_16k_negative'])
        logs['e'] = self.networks['Analysis'].energy(batch['gt_mel_22k'])
        logs['ps'] = self.networks['Analysis'].pitch.yingram_batch(batch['gt_audio_g'])
        logs['ps'] = logs['ps'][:, 19:69]

        result = self.networks['Synthesis'](logs['lps'], logs['s_pos'], logs['e'], logs['ps'])
        logs.update(result)

        loss['mel'] = F.l1_loss(logs['gen_mel'], logs['gt_mel_22k'])
        loss['backward'] = loss['mel']

        # for G
        if 'Discriminator' in self.networks.keys():
            pred_gen = self.networks['Discriminator'](logs['gen_mel'], logs['s_pos'], logs['s_neg'])
            loss['D_gen_forG'] = self.losses['BCE'](pred_gen, torch.ones_like(pred_gen))
            loss['backward'] = loss['backward'] + loss['D_gen_forG']

        # for D
        if 'Discriminator' in self.networks.keys():
            logs['gen_mel'] = logs['gen_mel'].detach()
            logs['s_pos'] = logs['s_pos'].detach()
            logs['s_neg'] = logs['s_neg'].detach()
            pred_gen = self.networks['Discriminator'](logs['gen_mel'], logs['s_pos'], logs['s_neg'])
            pred_gt = self.networks['Discriminator'](logs['gt_mel_22k'], logs['s_pos'], logs['s_neg'])
            loss['D_gen_forD'] = self.losses['BCE'](pred_gen, torch.zeros_like(pred_gen))
            loss['D_gt_forD'] = self.losses['BCE'](pred_gen, torch.ones_like(pred_gt))
            loss['D_backward'] = loss['D_gt_forD'] + loss['D_gen_forD']

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.common_step(batch, batch_idx)

        opts = self.optimizers()
        opts[self.opt_tag['Analysis']].zero_grad()
        opts[self.opt_tag['Synthesis']].zero_grad()

        loss['backward'].backward()

        if 'Discriminator' in self.networks.keys():
            opts[self.opt_tag['Discriminator']].zero_grad()
            loss['D_backward'].backward()

        opts[self.opt_tag['Analysis']].step()
        opts[self.opt_tag['Synthesis']].step()
        if 'Discriminator' in self.networks.keys():
            opts[self.opt_tag['Discriminator']].step()

        if self.global_step % self.conf.logging.freq == 0:
            self.awesome_logging(loss, mode='train')
            self.awesome_logging(logs, mode='train')

    def validation_step(self, batch, batch_idx):
        loss, logs = self.common_step(batch, batch_idx)
        self.awesome_logging(loss, mode='eval')
        self.awesome_logging(logs, mode='eval')

    def awesome_logging(self, data, mode):
        tensorboard = self.logger.experiment
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.squeeze()
                if value.ndim == 0:
                    tensorboard.add_scalar(f'{mode}/{key}', value, self.global_step)
                elif value.ndim == 3:
                    if value.shape[0] == 3:  # if 3-dim image
                        tensorboard.add_image(f'{mode}/{key}', value, self.global_step, dataformats='CHW')
                    else:  # B x H x W shaped images
                        value_numpy = value[0].detach().cpu().numpy()  # select one in batch
                        plt_image = self.plot_spectrogram_to_numpy(value_numpy)
                        tensorboard.add_image(f'{mode}/{key}', plt_image, self.global_step, dataformats='HWC')
                if 'audio' in key:
                    sample_rate = 22050
                    if '16k' in key:
                        sample_rate = 16000
                    tensorboard.add_audio(f'{mode}/{key}', value[0].unsqueeze(0), self.global_step,
                                          sample_rate=sample_rate)

            if isinstance(value, np.ndarray):
                if value.ndim == 3:
                    tensorboard.add_image(f'{mode}/{key}', value, self.global_step, dataformats='HWC')

    @staticmethod
    def plot_spectrogram_to_numpy(spectrogram):
        global MATPLOTLIB_FLAG
        if not MATPLOTLIB_FLAG:
            import matplotlib
            matplotlib.use("Agg")
            MATPLOTLIB_FLAG = True
        import matplotlib.pylab as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

    def test_time_self_adaptation(self, batch, batch_idx):
        loss = {}
        logs = {}
        logs.update(batch)

        logs['lps'] = self.networks['Analysis'].linguistic(batch['gt_audio_f'])
        logs['s_pos'] = self.networks['Analysis'].speaker(batch['gt_audio_16k'])
        logs['s_neg'] = self.networks['Analysis'].speaker(batch['gt_audio_16k_negative'])
        logs['e'] = self.networks['Analysis'].energy(batch['gt_mel_22k'])
        logs['ps'] = self.networks['Analysis'].pitch.yingram_batch(batch['gt_audio_g'])
        logs['ps'] = logs['ps'][:, 19:69]

        result = self.networks['Synthesis'](logs['lps'], logs['s_pos'], logs['e'], logs['ps'])
        logs.update(result)

        loss['mel'] = F.l1_loss(logs['gen_mel'], logs['gt_mel_22k'])
        loss['backward'] = loss['mel']

        opt = torch.optim.Adam(logs['lps'], lr=1e-4, betas=(0.5, 0.9))
        opt.zero_grad()
        loss['mel'].backward()
        opt.step()

        with torch.no_grad():
            logs['mel_filter'] = self.networks['Synthesis'](logs['lps'], logs['e'], logs['s_pos'])
            logs['gen_mel'] = logs['mel_filter'] + logs['mel_source']
            logs['audio_gen'] = self.networks['Synthesis'].vocoder(logs['gen_mel'])
        return loss, logs
