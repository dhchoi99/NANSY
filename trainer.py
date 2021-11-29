import importlib

hifigan = importlib.import_module('models.hifi-gan')
hifigan_vocoder = getattr(hifigan, 'Generator')
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
import pytorch_lightning as pl

MATPLOTLIB_FLAG = False


class Trainer(pl.LightningModule):
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.conf = conf

        self.networks = torch.nn.ModuleDict(self.build_models())

        self.opt_tag = {key: None for key in self.networks.keys()}

        self.losses = self.build_losses()

        # self.load_wav2vec2()
        # self.load_vocoder()

    def load_wav2vec2(self):
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53")
        self.wav2vec2.eval()
        self.wav2vec2 = self.wav2vec2.to(self.device)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def load_vocoder(self):
        path_config = './configs/hifi-gan/config.json'
        with open(path_config) as f:
            data = f.read()
        json_config = json.loads(data)

        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        hifigan_config = AttrDict(json_config)
        self.vocoder = hifigan_vocoder(hifigan_config)

        path_ckpt = './configs/hifi-gan/generator_v1'

        def load_checkpoint(filepath):
            assert os.path.isfile(filepath)
            print("Loading '{}'".format(filepath))
            checkpoint_dict = torch.load(filepath)
            print("Complete.")
            return checkpoint_dict

        state_dict_g = load_checkpoint(path_ckpt)
        self.vocoder.load_state_dict(state_dict_g['generator'])
        self.vocoder.eval()
        self.vocoder = self.vocoder.to(self.device)
        for param in self.vocoder.parameters():
            param.requires_grad = False

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

        logs['lps'] = self.networks['Analysis'].linguistic(batch['gt_audio_f'])
        # with torch.no_grad():
        # wav2vec2_output = self.wav2vec(batch['gt_audio_f'], output_hidden_states=True)
        # logs['lps'] = wav2vec2_output.hidden_states[1].permute((0, 2, 1))  # B x C x t

        # wav2vec2_output = self.wav2vec2(batch['gt_audio_16k'], output_hidden_states=True)
        # s_pos_pre = wav2vec2_output.hidden_states[12].permute((0, 2, 1))  # B x C x t
        # wav2vec2_output = self.wav2vec2(batch['gt_audio_16k_negative'], output_hidden_states=True)
        # s_neg_pre = wav2vec2_output.hidden_states[12].permute((0, 2, 1))  # B x C x t

        logs['s_pos'] = self.networks['Analysis'].speaker(batch['gt_audio_16k'])
        logs['s_neg'] = self.networks['Analysis'].speaker(batch['gt_audio_16k_negative'])
        # logs['s_pos'] = self.networks['Analysis'].speaker(s_pos_pre)
        # logs['s_neg'] = self.networks['Analysis'].speaker(s_neg_pre)

        with torch.no_grad():
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
            # loss['D_gen_forG'] = self.losses['BCE'](pred_gen, torch.ones_like(pred_gen))
            loss['D_gen_forG'] = torch.mean(torch.sigmoid(pred_gen))
            loss['backward'] = loss['backward'] + loss['D_gen_forG']

        # for D
        if 'Discriminator' in self.networks.keys():
            logs['gen_mel'] = logs['gen_mel'].detach()
            logs['s_pos'] = logs['s_pos'].detach()
            logs['s_neg'] = logs['s_neg'].detach()
            pred_gen = self.networks['Discriminator'](logs['gen_mel'], logs['s_pos'], logs['s_neg'])
            pred_gt = self.networks['Discriminator'](logs['gt_mel_22k'], logs['s_pos'], logs['s_neg'])
            # loss['D_gen_forD'] = self.losses['BCE'](pred_gen, torch.zeros_like(pred_gen))
            loss['D_gen_forD'] = -torch.mean(torch.sigmoid(pred_gen))
            # if not torch.isfinite(loss['D_gen_forD']):
            #     raise AssertionError('D_gen_forD')
            # loss['D_gt_forD'] = self.losses['BCE'](pred_gt, torch.ones_like(pred_gt))
            loss['D_gt_forD'] = torch.mean(torch.sigmoid(pred_gt))
            # if not torch.isfinite(loss['D_gt_forD']):
            #     raise AssertionError('D_gt_forD')
            loss['D_backward'] = loss['D_gt_forD'] + loss['D_gen_forD']

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
        self.awesome_logging(loss, mode='eval')
        self.awesome_logging(logs, mode='eval')

    def awesome_logging(self, data, mode):
        tensorboard = self.logger.experiment
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.squeeze()
                if value.ndim == 0:
                    # tensorboard.add_scalar(f'{mode}/{key}', value, self.global_step)
                    self.log(f'{mode}/{key}', value)
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
