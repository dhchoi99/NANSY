import importlib

import matplotlib.pylab as plt
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.functional as AF
import transformers
import pytorch_lightning as pl

from models.loss import GANLoss
from models.hifi_gan import Generator as hifigan_vocoder

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
        conf_ganloss = DictConfig({
            'gan_mode': 'lsgan',
            'real': 1.0,
            'fake': 0.0,
        })
        losses_dict['GANLoss'] = GANLoss(conf_ganloss)

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

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if name != 'wav2vec2' and name != 'vocoder':
                module.train(mode)
        return self

    def common_step(self, batch, batch_idx):
        loss = {}
        logs = {}
        logs.update(batch)

        logs['lps'] = self.networks['Analysis'].linguistic(batch['gt_audio_16k_f'])

        logs['s_pos'] = self.networks['Analysis'].speaker(batch['gt_audio_16k'])
        logs['s_neg'] = self.networks['Analysis'].speaker(batch['gt_audio_16k_negative'])
        # logs['s_pos'] = self.networks['Analysis'].speaker(s_pos_pre)
        # logs['s_neg'] = self.networks['Analysis'].speaker(s_neg_pre)

        # non-training calculations
        with torch.no_grad():
            logs['e'] = self.networks['Analysis'].energy(batch['gt_mel_22k'])
            logs['ps'] = self.networks['Analysis'].pitch.yingram_batch(batch['gt_audio_22k_g'])
            logs['ps'] = logs['ps'][:, 19:69]

        result = self.networks['Synthesis'](logs['lps'], logs['s_pos'], logs['e'], logs['ps'])
        logs.update(result)

        loss['mel'] = F.l1_loss(logs['gen_mel'], logs['gt_mel_22k'])
        loss['backward'] = loss['mel']

        # for G
        if 'Discriminator' in self.networks.keys():
            pred_gen = self.networks['Discriminator'](logs['gen_mel'], logs['s_pos'], logs['s_neg'])

            loss['D_gen_forG'] = self.losses['GANLoss'](pred_gen, True, False)
            # loss['D_gen_forG'] = self.losses['BCE'](pred_gen, torch.ones_like(pred_gen))
            # loss['D_gen_forG'] = torch.mean(torch.sigmoid(pred_gen)) # 0=gt, 1=gen
            loss['backward'] = loss['backward'] + 1 * loss['D_gen_forG']

        # for D
        if 'Discriminator' in self.networks.keys():
            logs['gen_mel'] = logs['gen_mel'].detach()
            logs['s_pos'] = logs['s_pos'].detach()
            logs['s_neg'] = logs['s_neg'].detach()
            pred_gen = self.networks['Discriminator'](logs['gen_mel'], logs['s_pos'], logs['s_neg'])
            pred_gt = self.networks['Discriminator'](logs['gt_mel_22k'], logs['s_pos'], logs['s_neg'])

            loss['D_gen_forD'] = self.losses['GANLoss'](pred_gen, False, True)
            loss['D_gt_forD'] = self.losses['GANLoss'](pred_gt, True, True)
            loss['D_backward'] = loss['D_gen_forD'] + loss['D_gt_forD']

            # loss['D_gen_forD'] = self.losses['BCE'](pred_gen, torch.zeros_like(pred_gen))
            # loss['D_gt_forD'] = self.losses['BCE'](pred_gt, torch.ones_like(pred_gt))
            # loss['D_backward'] = 1 * loss['D_gt_forD'] + loss['D_gen_forD']

            # loss['D_gen_forD'] = torch.mean(torch.sigmoid(pred_gen))
            # loss['D_gt_forD'] = torch.mean(torch.sigmoid(pred_gt))
            # loss['D_backward'] = 1 * (loss['D_gt_forD'] - loss['D_gen_forD'])

        # reconstruction loss
        # gen_audio_22k = result['audio_gen']
        # gen_audio_16k = AF.resample(gen_audio_22k, 22050, 16000)
        #
        # logs['recon_lps'] = self.networks['Analysis'].linguistic(gen_audio_16k)
        # logs['recon_s'] = self.networks['Analysis'].speaker(gen_audio_16k)
        #
        # loss['recon_lps'] = self.losses['L1'](logs['recon_lps'], logs['lps'])
        # loss['recon_s'] = self.losses['L1'](logs['recon_s'], logs['s_pos'])
        # loss['recon'] = loss['recon_lps'] + loss['recon_s']
        # loss['backward'] = loss['backward'] + loss['recon']

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.common_step(batch, batch_idx)

        opts = self.optimizers()
        opts[self.opt_tag['Analysis']].zero_grad()
        opts[self.opt_tag['Synthesis']].zero_grad()

        # loss['backward'].backward()
        self.manual_backward(loss['backward'])

        if 'Discriminator' in self.networks.keys():
            opts[self.opt_tag['Discriminator']].zero_grad()
            # loss['D_backward'].backward()
            self.manual_backward(loss['D_backward'])

        # for model in self.networks.values():
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        opts[self.opt_tag['Analysis']].step()
        opts[self.opt_tag['Synthesis']].step()
        if 'Discriminator' in self.networks.keys():
            opts[self.opt_tag['Discriminator']].step()

        if self.global_step % self.conf.logging.freq == 0:
            self.awesome_logging(loss, mode='train')
            self.awesome_logging(logs, mode='train')

    def validation_step(self, batch, batch_idx):
        loss, logs = self.common_step(batch, batch_idx)
        if batch_idx == 0:
            self.awesome_logging(loss, mode='eval')
            self.awesome_logging(logs, mode='eval')

    def awesome_logging(self, data, mode):
        tensorboard = self.logger.experiment
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.squeeze()
                if value.ndim == 0:
                    # tensorboard.add_scalar(f'{mode}/{key}', value, self.global_step)
                    self.log(f'{mode}/{key}', value, batch_size=self.conf.datasets.train.batch_size)
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
