import argparse
import glob
import json
import os
import importlib

hifigan = importlib.import_module('models.hifi-gan')
hifigan_vocoder = getattr(hifigan, 'Generator')

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm, trange

from utils.util import save_files, build_models_from_config, build_datasets_from_config
from utils.logging import get_logger

MATPLOTLIB_FLAG = False


class Trainer:
    def __init__(self, conf):
        super(Trainer, self).__init__()
        self.conf = conf
        self.device = self.conf.logging.device
        self.save_files()

        self.build_datasets()
        self.build_models()
        self.build_losses()

        self.set_logger()

        self.load_wav2vec2()
        self.load_vocoder()

    def load_wav2vec2(self):
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        self.wav2vec2.eval()
        self.wav2vec2 = self.wav2vec2.to(self.device)

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

    # region init
    def save_files(self):
        savefiles = []
        for glob_path in self.conf.logging['save_files']:
            savefiles += glob.glob(glob_path)
        save_files(self.conf.logging['log_dir'], savefiles)

    def build_datasets(self):
        datasets, loaders, iterators = build_datasets_from_config(self.conf.datasets)
        self.datasets = datasets
        self.loaders = loaders
        self.iterators = iterators

    def build_models(self):
        models, optims = build_models_from_config(self.conf.models)

        for key, value in models.items():
            models[key] = value.to(self.device)
        self.models = models
        self.optims = optims

    def build_losses(self):
        losses_dict = {}
        losses_dict['L1'] = torch.nn.L1Loss()
        losses_dict['BCE'] = torch.nn.BCEWithLogitsLoss()
        self.losses = losses_dict

    def set_logger(self):
        logger = get_logger(self.conf.logging)
        self.logger = logger
        self.global_step = 0
        self.global_epoch = 0

    # endregion

    # region training

    def forward_g(self, batch):
        result = {}

        with torch.no_grad():
            wav2vec2_output = self.wav2vec2(batch['gt_audio_f'], output_hidden_states=True)
            lps = wav2vec2_output.hidden_states[1].permute((0, 2, 1))  # B x C x t
            result['lps'] = lps

        with torch.no_grad():
            wav2vec2_output = self.wav2vec2(batch['gt_audio_16k'], output_hidden_states=True)
            s_pos_pre = wav2vec2_output.hidden_states[12].permute((0, 2, 1))  # B x C x t
            wav2vec2_output = self.wav2vec2(batch['gt_audio_16k_negative'], output_hidden_states=True)
            s_neg_pre = wav2vec2_output.hidden_states[12].permute((0, 2, 1))  # B x C x t

        s_pos = self.models['Analysis'].speaker(s_pos_pre)
        s_neg = self.models['Analysis'].speaker(s_neg_pre)

        result['s_pos'] = s_pos
        result['s_neg'] = s_neg

        with torch.no_grad():
            e = self.models['Analysis'].energy(batch['gt_mel_22k'])
            ps = self.models['Analysis'].pitch.yingram_batch(batch['gt_audio_g'])
            ps = ps[:, 19:69]

        result['e'] = e
        result['ps'] = ps

        output = self.models['Synthesis'](lps, s_pos, e, ps)
        result.update(output)

        result['audio_gen'] = self.vocoder(result['gen_mel'])

        return result

    def train_step(self, batch):
        loss = {}
        logs = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logs[key] = value.cpu().detach()
                batch[key] = value.to(self.device)
            else:
                logs[key] = value

        # g_step
        for key in self.models.keys():
            if 'Discriminator' in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        result = self.forward_g(batch)
        loss['mel'] = F.l1_loss(result['gen_mel'], batch['gt_mel_22k'])
        loss['backward'] = loss['mel']

        if 'Discriminator' in self.models.keys():
            pred_gen = self.models['Discriminator'](result['gen_mel'], result['s_pos'], result['s_neg'])
            loss['D_gen_forG'] = self.losses['BCE'](pred_gen, torch.ones_like(pred_gen))
            loss['backward'] = loss['backward'] + loss['D_gen_forG']

        self.optims['Analysis'].zero_grad()
        self.optims['Synthesis'].zero_grad()
        loss['backward'].backward()
        self.optims['Analysis'].step()
        self.optims['Synthesis'].step()

        # d_step
        for key in self.models.keys():
            if 'Discriminator' not in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        if 'Discriminator' in self.models.keys():
            gen_mel = result['gen_mel'].detach()
            s_pos = result['s_pos'].detach()
            s_neg = result['s_neg'].detach()

            pred_gen = self.models['Discriminator'](gen_mel, s_pos, s_neg)
            pred_gt = self.models['Discriminator'](batch['gt_mel_22k'], s_pos, s_neg)
            loss['D_gen_forD'] = self.losses['BCE'](pred_gen, torch.zeros_like(pred_gen))
            loss['D_gt_forD'] = self.losses['BCE'](pred_gt, torch.ones_like(pred_gt))
            loss['D_backward'] = loss['D_gt_forD'] + loss['D_gen_forD']

        self.optims['Discriminator'].zero_grad()
        loss['D_backward'].backward()
        self.optims['Discriminator'].step()

        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                logs[key] = value.cpu().detach()

        return loss, logs

    def eval_step(self, batch):
        loss = {}
        logs = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                logs[key] = value.cpu().detach()
                batch[key] = value.to(self.device)
            else:
                logs[key] = value

        # g_step
        for key in self.models.keys():
            if 'Discriminator' in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        result = self.forward_g(batch)
        loss['mel'] = F.l1_loss(result['gen_mel'], batch['gt_mel_22k'])
        loss['backward'] = loss['mel']

        if 'Discriminator' in self.models.keys():
            pred_gen = self.models['Discriminator'](result['gen_mel'], result['s_pos'], result['s_neg'])
            loss['D_gen_forG'] = self.losses['BCE'](pred_gen, torch.ones_like(pred_gen))
            loss['backward'] = loss['backward'] + loss['D_gen_forG']

        # d_step
        for key in self.models.keys():
            if 'Discriminator' not in self.models.keys():
                self.models['Discriminator'].eval()
            else:
                self.models[key].train()

        if 'Discriminator' in self.models.keys():
            gen_mel = result['gen_mel'].detach()
            s_pos = result['s_pos'].detach()
            s_neg = result['s_neg'].detach()

            pred_gen = self.models['Discriminator'](gen_mel, s_pos, s_neg)
            pred_gt = self.models['Discriminator'](batch['gt_mel_22k'], s_pos, s_neg)
            loss['D_gen_forD'] = self.losses['BCE'](pred_gen, torch.zeros_like(pred_gen))
            loss['D_gt_forD'] = self.losses['BCE'](pred_gt, torch.ones_like(pred_gt))
            loss['D_backward'] = loss['D_gt_forD'] + loss['D_gen_forD']

        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                logs[key] = value.cpu().detach()

        return loss, logs

    def train_epoch(self):
        pbar_step = trange(len(self.loaders['train']), position=1)
        pbar_step.set_description_str('STEP')

        losses_train = {}
        losses_eval = {}

        for step in pbar_step:
            train_data = next(self.iterators['train'])
            loss_train, log_train = self.train_step(train_data)
            if self.global_step % self.conf.logging.freq == 0:
                self.awesome_logging(loss_train, mode='train')
                self.awesome_logging(log_train, mode='train')
            for key, value in loss_train.items():
                if key not in losses_train.keys():
                    losses_train[key] = 0.
                losses_train[key] += value.data

            if self.global_step % self.conf.logging.freq == 0:
                eval_data = next(self.iterators['eval'])
                with torch.no_grad():
                    loss_eval, log_eval = self.eval_step(eval_data)
                    for key, value in loss_eval.items():
                        if key not in losses_eval.keys():
                            losses_eval[key] = 0.
                        losses_eval[key] += value.data

                self.awesome_logging(loss_eval, mode='eval')
                self.awesome_logging(log_eval, mode='eval')

            self.global_step += 1

        # losses_train = {
        #     key: value / len(pbar_step)
        #     for key, value in losses_train.items()
        # }
        # losses_eval = {
        #     key: value / (len(pbar_step) // self.conf.logging.freq['eval'])
        #     for key, value in losses_eval.items()
        # }
        # self.log(losses_train, {}, mode='train', key='epoch')
        # self.log(losses_eval, {}, mode='eval', key='epoch')

    def run(self):
        pbar_epoch = trange(self.conf.logging.nepochs, position=0)
        pbar_epoch.set_description_str('Epoch')

        for epoch in pbar_epoch:
            self.train_epoch()
            self.global_epoch += 1
            self.save()

    # endregion

    def save(self):
        models = {}
        for key in self.models.keys():
            models[key] = {}
            models[key]['state_dict'] = self.models[key].state_dict()
            if self.conf.logging.save_optimizer_state:
                models[key]['optimizer'] = self.optims[key].state_dict()

        dir_save = os.path.join(self.conf.logging.log_dir, 'checkpoint')
        os.makedirs(dir_save, exist_ok=True)
        path_save = os.path.join(dir_save, f'step_{self.global_step}.pth')

        torch.save(models, path_save, _use_new_zipfile_serialization=False)

    def awesome_logging(self, data, mode):
        tensorboard = self.logger  # TODO
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


def main():
    args = parse_args()
    conf = OmegaConf.load(args.config)

    conf.logging['log_dir'] = os.path.join(conf.logging['log_dir'], str(conf.logging['seed']))
    os.makedirs(conf.logging['log_dir'], exist_ok=True)

    if args.train:
        trainer = Trainer(conf)
        trainer.run()

    if args.infer:
        # inference()
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_nansy.yaml',
                        help='config file path')

    parser.add_argument('--train', action='store_true',
                        help='')
    parser.add_argument('--infer', action='store_true',
                        help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
