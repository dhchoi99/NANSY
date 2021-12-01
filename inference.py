#!/usr/bin/env python
# coding: utf-8


import argparse
from collections import OrderedDict

import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
import torchaudio.functional as AF
from tqdm.notebook import tqdm, trange

from datasets.custom import CustomDataset
from models.analysis import Analysis
from models.synthesis import Synthesis


def pl_checkpoint_to_torch_checkpoints(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        module_name = key.split('.')[1]
        if module_name not in new_state_dict.keys():
            new_state_dict[module_name] = OrderedDict()

        new_key = key.split('.', 2)[-1]
        new_state_dict[module_name][new_key] = state_dict[key]

    return new_state_dict


class TSAHelper(torch.nn.Module):
    def __init__(self):
        super(TSAHelper, self).__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(1, 1024, 74).float())

    def forward(self, x):
        y = x + self.embedding
        return y


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_audio_conf', type=str, default='configs/audio/22k.yaml',
                        help='')
    parser.add_argument('--path_ckpt', type=str, required=True,
                        help='path to pl checkpoint')
    parser.add_argument('--path_audio_source', type=str, required=True,
                        help='path to source audio file, sr=22k')

    parser.add_argument('--path_audio_target', type=str, required=True,
                        help='path to target audio file, sr=16k')

    parser.add_argument('--tsa_loop', type=int, default=100,
                        help='iterations for tsa')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # path_audio_conf = 'configs/audio/22k.yaml'
    # path_ckpt = '/raid/vision/dhchoi/log/nansy/31/checkpoints/epoch=33-step=154708.ckpt'
    # path_audio_source = '/raid/vision/dhchoi/data/VCTK-Corpus/wav22/p376/p376_293-22k.wav'
    # path_audio_target = '/raid/vision/dhchoi/temp/DS2632_00322.wav'
    # tsa_loop = 100

    data_ckpt = torch.load(args.path_ckpt, map_location='cpu')
    state_dict = data_ckpt['state_dict']

    new_state_dict = pl_checkpoint_to_torch_checkpoints(state_dict)

    synthesis = Synthesis(None)
    analysis = Analysis(None)

    analysis.load_state_dict(new_state_dict['Analysis'])
    synthesis.load_state_dict(new_state_dict['Synthesis'])

    analysis.cuda(1).eval()
    synthesis.cuda(1).eval()

    # for param in analysis.parameters():
    #     param.requires_grad = False
    #
    # for key, param in synthesis.named_parameters():
    #     if key.startswith('filter_generator'):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    networks = {'Analysis': analysis, 'Synthesis': synthesis}

    tsa_helper = TSAHelper().cuda(1)
    print(tsa_helper.embedding)
    opt = torch.optim.Adam(tsa_helper.parameters(), lr=1e-4, betas=(0.5, 0.9))

    conf_audio = OmegaConf.load(args.path_audio_conf)
    conf = DictConfig({'audio': conf_audio})

    self = CustomDataset(conf)
    _, wav_22k_source = self.load_wav(args.path_audio_source, 22050)
    wav_16k_source = AF.resample(wav_22k_source, 22050, 16000)
    _, wav_16k_target = self.load_wav(args.path_audio_target, 16000)
    mel_22k = self.load_mel(args.path_audio_source, sr=22050)

    return_data = {}

    audios = {
        'gt_source': [],
        'gt_target': [],
        'gen_source': [],
        'gen_target': [],
        'gen_source_tsa': [],
        'gen_target_tsa': [],
    }

    for idx in trange(0, mel_22k.shape[-1], self.mel_window):
        mel_start = idx
        pos_time_idxs = self.get_time_idxs(mel_start)

        gt_mel_22k = self.crop_audio(mel_22k, pos_time_idxs[0], pos_time_idxs[1], -self.mel_padding_value)

        source_16k = self.crop_audio(wav_16k_source, pos_time_idxs[3], pos_time_idxs[5])
        target_16k = self.crop_audio(wav_16k_target, pos_time_idxs[3], pos_time_idxs[5])
        source_22k = self.crop_audio(wav_22k_source, pos_time_idxs[4], pos_time_idxs[6])
        source_22k_yin = self.crop_audio(wav_22k_source, pos_time_idxs[4], pos_time_idxs[7])

        return_data['source_22k'] = source_22k
        return_data['source_16k'] = source_16k

        return_data['target_16k'] = target_16k

        return_data['source_22k_yin'] = source_22k_yin
        return_data['source_22k_mel'] = gt_mel_22k

        batch = {key: value.unsqueeze(0).cuda(1) for key, value in return_data.items()}

        audios['gt_source'].append(batch['source_22k'][0].cpu().numpy())
        audios['gt_target'].append(batch['target_16k'][0].cpu().numpy())

        with torch.no_grad():
            lps = networks['Analysis'].linguistic(batch['source_16k'])
            s_src = networks['Analysis'].speaker(batch['source_16k'])
            s_tgt = networks['Analysis'].speaker(batch['target_16k'])
            e = networks['Analysis'].energy(batch['source_22k_mel'])
            ps = networks['Analysis'].pitch.yingram_batch(batch['source_22k'])
            ps = ps[:, 19:69]

        with torch.no_grad():
            result = networks['Synthesis'](lps, s_src, e, ps)
            audios['gen_source'].append(result['audio_gen'][0].cpu().numpy())

        with torch.no_grad():
            result = networks['Synthesis'](lps, s_tgt, e, ps)
            audios['gen_target'].append(result['audio_gen'][0].cpu().numpy())

        for i in range(args.tsa_loop):  # test time self adaptation
            lps_tsa = lps.detach().clone()
            lps_tsa = tsa_helper(lps_tsa)
            result = networks['Synthesis'](lps_tsa, s_src, e, ps)

            opt.zero_grad()
            loss_mel = torch.nn.functional.l1_loss(batch['source_22k_mel'], result['gen_mel'])
            loss_mel.backward()
            opt.step()

        with torch.no_grad():
            lps = tsa_helper(lps)
            result = networks['Synthesis'](lps, s_src, e, ps)
            audios['gen_source_tsa'].append(result['audio_gen'][0].cpu().numpy())
            result = networks['Synthesis'](lps, s_tgt, e, ps)
            audios['gen_target_tsa'].append(result['audio_gen'][0].cpu().numpy())

    print(tsa_helper.embedding)

    final_audios = {}
    for key, value in audios.items():
        try:
            final_audios[key] = np.concatenate(value, axis=-1)
            print(key, len(value), final_audios[key].shape)
        except Exception as e:
            print(key, e)

    return final_audios


if __name__ == '__main__':
    import os
    from scipy.io import wavfile

    final_audios = main()

    dir_save = './temp_result'
    os.makedirs(dir_save, exist_ok=True)
    for key, value in final_audios.items():
        sample_rate = 22050 if key != 'gt_target' else 16000
        try:
            wavfile.write(
                os.path.join(dir_save, f'{key}.wav'),
                sample_rate,
                value.squeeze().astype(np.float32))
        except Exception as e:
            print(key, e)

    # import IPython.display as ipd
    # ipd.Audio(final_audios['gt_source'], rate=22050)
    # ipd.Audio(final_audios['gt_target'], rate=16000)
    # ipd.Audio(final_audios['gen_source'], rate=22050)
    # ipd.Audio(final_audios['gen_target'], rate=22050)
    # ipd.Audio(final_audios['gen_source_tsa'], rate=22050)
    # ipd.Audio(final_audios['gen_target_tsa'], rate=22050)
