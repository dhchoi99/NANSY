import argparse
import glob
import os

import numpy as np
from omegaconf import OmegaConf
import torch
import torchaudio.functional as AF

from datasets.base import MultiDataset
from trainer import Trainer


class TSAHelper(torch.nn.Module):
    def __init__(self):
        super(TSAHelper, self).__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(1, 1024, 73).float())

    def forward(self, x):
        y = x + self.embedding
        return y


def main():
    path_conf = 'configs/train_nansy.yaml'
    conf = OmegaConf.load(path_conf)

    model = Trainer(conf)  # TODO get trainer from conf
    model.eval()

    path_ckpt = '/raid/vision/dhchoi/log/nansy/16/checkpoints/epoch=47-step=52367.ckpt'
    data_ckpt = torch.load(path_ckpt, map_location='cpu')

    model.load_state_dict(data_ckpt['state_dict'])

    path_audio_source = '/raid/vision/dhchoi/data/VCTK-Corpus/wav22/p376/p376_290-22k.wav'
    path_audio_target = '/raid/vision/dhchoi/temp/DS2632_00322.wav'

    d = MultiDataset(conf.datasets.train)
    tsa_helper = TSAHelper()

    opt = torch.optim.Adam(tsa_helper.parameters(), lr=1e-4, betas=(0.5, 0.9))

    _, wav_22k_source = d.datasets['0'].load_wav(path_audio_source, 22050)
    wav_16k_source = AF.resample(wav_22k_source, 22050, 16000)
    _, wav_16k_target = d.datasets['0'].load_wav(path_audio_target, 16000)
    mel_22k = d.datasets['0'].load_mel(path_audio_source, sr=22050)

    self = d.datasets['0']
    return_data = {}
    # for idx in range(0, mel_22k.shape[1-1].shape[-1]-1, self.mel_len):

    audios = {
        'gt_source': [],
        'gt_target': [],
        'gen_source': [],
        'gen_target': [],
        'gen_source_tsa': [],
        'gen_target_tsa': [],
    }
    for idx in range(0, mel_22k.length + d.datasets['0'].mel_len, d.datasets['0'].mel_len):
        mel_start = idx
        mel_end = mel_start + self.mel_len
        gt_mel_22k = self.crop_audio(mel_22k, mel_start, mel_end, -4)

        t_start = mel_start * self.conf.audio.hop_size / 22050.
        w_start_22k = int(t_start * 22050)
        w_start_16k = int(t_start * 16000)
        w_end_22k = w_start_22k + self.window_size
        w_end_22k_yin = w_start_22k + self.yin_window_size
        w_end_16k = w_start_16k + self.window_size_16k

        source_22k = self.crop_audio(wav_22k_source, w_start_22k, w_end_22k)
        source_16k = self.crop_audio(wav_16k_source, w_start_16k, w_end_16k)
        target_16k = self.crop_audio(wav_16k_target, w_start_16k, w_end_16k)
        source_22k_yin = self.crop_audio(wav_22k_source, w_start_22k, w_end_22k_yin)

        return_data['source_22k'] = source_22k
        return_data['source_16k'] = source_16k

        return_data['target_16k'] = target_16k

        return_data['source_22k_yin'] = source_22k_yin
        return_data['source_22k_mel'] = gt_mel_22k

        #     return_data['t'] = t_start
        batch = {key: value.unsqueeze(0) for key, value in return_data.items()}

        audios['gt_source'].append(batch['source_22k'][0].cpu().numpy())

        with torch.no_grad():
            lps = model.networks['Analysis'].linguistic(batch['source_16k'])
            s_src = model.networks['Analysis'].speaker(batch['source_16k'])
            s_tgt = model.networks['Analysis'].speaker(batch['target_16k'])
            e = model.networks['Analysis'].energy(batch['source_22k_mel'])
            ps = model.networks['Analysis'].pitch.yingram_batch(batch['source_22k'])
            ps = ps[:, 19:69]

        with torch.no_grad():
            result = model.networks['Synthesis'](lps, s_src, e, ps)
            audios['gen_source'].append(result['audio_gen'][0].cpu().numpy())

        with torch.no_grad():
            result = model.networks['Synthesis'](lps, s_tgt, e, ps)
            audios['gen_target'].append(result['audio_gen'][0].cpu().numpy())

        for i in range(100):  # test time self adaptation
            lps = tsa_helper(lps)
            result = model.networks['Synthesis'](lps, s_src, e, ps)

            opt.zero_grad()
            loss_mel = torch.nn.functional.l1_loss(batch['source_22k_mel'], result['gt_mel_22k'])
            loss_mel.backward()
            opt.step()

        with torch.no_grad():
            lps = tsa_helper(lps)
            result = model.networks['Synthesis'](lps, s_src, e, ps)
            audios['gen_source_tsa'].append(result['audio_gen'][0].cpu().numpy())
            result = model.networks['Synthesis'](lps, s_tgt, e, ps)
            audios['gen_target_tsa'].append(result['audio_gen'][0].cpu().numpy())

    final_audios = {}
    for key, value in audios.items():
        final_audios[key] = np.stack(value, axis=0)


if __name__ == '__main__':
    main()