import argparse
from collections import OrderedDict
import math

import numpy as np
from omegaconf import OmegaConf, DictConfig
import parselmouth
import torch
import torchaudio.functional as AF
from tqdm.notebook import tqdm, trange

from datasets.custom import CustomDataset
from models.analysis import Analysis
from models.synthesis import Synthesis
from datasets.functional import wav_to_Sound


def hz_diff_to_midi_diff(a, b, semitone_range=12):
    ratio = a / b
    ratio_linear = math.log(ratio, 2)
    midi_diff = semitone_range * ratio_linear
    return midi_diff


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

    parser.add_argument('--device', type=str, default='cuda',
                        help='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    conf_audio = OmegaConf.load(args.path_audio_conf)
    conf = DictConfig({'audio': conf_audio})
    self = CustomDataset(conf)

    _, wav_22k_a_torch = self.load_wav(args.path_audio_source, 22050)
    wav_16k_a_torch = AF.resample(wav_22k_a_torch, 22050, 16000)
    mel_22k_a = self.load_mel(args.path_audio_source, sr=22050)

    _, wav_16k_b_torch = self.load_wav(args.path_audio_target, 16000)
    wav_22k_b_torch = AF.resample(wav_16k_b_torch, 16000, 22050)
    mel_22k_b = self.load_mel(args.path_audio_target, sr=16000)

    data_ckpt = torch.load(args.path_ckpt, map_location='cpu')
    state_dict = data_ckpt['state_dict']
    new_state_dict = pl_checkpoint_to_torch_checkpoints(state_dict)

    analysis = Analysis(None)
    analysis.load_state_dict(new_state_dict['Analysis'])
    analysis.to(args.device).eval()

    synthesis = Synthesis(None)
    synthesis.load_state_dict(new_state_dict['Synthesis'])
    synthesis.to(args.device).eval()

    networks = {'Analysis': analysis, 'Synthesis': synthesis}

    # region analysis

    logs = {
        'lps_a': [],
        'lps_b': [],
        's_a': [],
        's_b': [],
        'e_a': [],
        'e_b': [],
        'ps_a': [],
        'ps_b': [],
        'a_mel_22k': [],
        'b_mel_22k': [],
    }

    audios = {
        'gt_a': [],
        'gt_b': [],
    }

    for idx in trange(0, mel_22k_a.shape[-1], self.mel_window):
        mel_start = idx
        pos_time_idxs = self.get_time_idxs(mel_start)

        gt_a_16k = self.crop_audio(wav_16k_a_torch, pos_time_idxs[3], pos_time_idxs[5])
        gt_a_22k = self.crop_audio(wav_22k_a_torch, pos_time_idxs[4], pos_time_idxs[6])
        gt_a_22k_yin = self.crop_audio(wav_22k_a_torch, pos_time_idxs[4], pos_time_idxs[7])
        gt_a_mel_22k = self.crop_audio(mel_22k_a, pos_time_idxs[0], pos_time_idxs[1],
                                       padding_value=self.mel_padding_value)

        gt_b_16k = self.crop_audio(wav_16k_b_torch, pos_time_idxs[3], pos_time_idxs[5])
        gt_b_22k = self.crop_audio(wav_22k_b_torch, pos_time_idxs[4], pos_time_idxs[6])
        gt_b_22k_yin = self.crop_audio(wav_22k_b_torch, pos_time_idxs[4], pos_time_idxs[7])
        gt_b_mel_22k = self.crop_audio(mel_22k_b, pos_time_idxs[0], pos_time_idxs[1],
                                       padding_value=self.mel_padding_value)

        return_data = {
            'a_wav_16k': gt_a_16k,
            'a_wav_22k': gt_a_22k,
            'a_wav_22k_yin': gt_a_22k_yin,
            'a_mel_22k': gt_a_mel_22k,

            'b_wav_16k': gt_b_16k,
            'b_wav_22k': gt_b_22k,
            'b_wav_22k_yin': gt_b_22k_yin,
            'b_mel_22k': gt_b_mel_22k,
        }

        batch = {key: value.unsqueeze(0).to(args.device) for key, value in return_data.items()}
        audios['gt_a'].append(batch['a_wav_22k'][0].cpu().numpy())
        audios['gt_b'].append(batch['b_wav_22k'][0].cpu().numpy())

        logs['a_mel_22k'].append(batch['a_mel_22k'])
        logs['b_mel_22k'].append(batch['b_mel_22k'])

        with torch.no_grad():
            lps_a = networks['Analysis'].linguistic(batch['a_wav_16k'])
            lps_b = networks['Analysis'].linguistic(batch['b_wav_16k'])

            logs['lps_a'].append(lps_a)
            logs['lps_b'].append(lps_b)

            s_a = networks['Analysis'].speaker(batch['a_wav_16k'])
            s_b = networks['Analysis'].speaker(batch['b_wav_16k'])
            logs['s_a'].append(s_a)
            logs['s_b'].append(s_b)

            e_a = networks['Analysis'].energy(batch['a_mel_22k'])
            e_b = networks['Analysis'].energy(batch['b_mel_22k'])

            logs['e_a'].append(e_a)
            logs['e_b'].append(e_b)

            ps_a_ori = networks['Analysis'].pitch.yingram_batch(batch['a_wav_22k_yin'])
            logs['ps_a'].append(ps_a_ori)
            ps_b_ori = networks['Analysis'].pitch.yingram_batch(batch['b_wav_22k_yin'])
            logs['ps_b'].append(ps_b_ori)

    # endregion

    # region synthesis

    audios.update({
        'recon_a': [],
        'recon_b': [],
        'text_a_spk_b': [],
        'text_b_spk_a': [],
        'recon_a_tsa': [],
        'text_a_spk_b_tsa': [],
    })
    tsa_helper = TSAHelper().to(args.device)
    opt = torch.optim.Adam(tsa_helper.parameters(), lr=1e-4, betas=(0.5, 0.9))
    pitch_median_a = None
    pitch_median_b = None

    for idx in range(len(logs['lps_a'])):
        lps_a = logs['lps_a'][idx]
        lps_b = logs['lps_b'][idx]

        s_a = logs['s_a'][idx]
        s_b = logs['s_b'][idx]

        e_a = logs['e_a'][idx]
        e_b = logs['e_b'][idx]

        ps_a_ori = logs['ps_a'][idx]
        ps_b_ori = logs['ps_b'][idx]

        # recon
        with torch.no_grad():
            result_a = networks['Synthesis'](lps_a, s_a, e_a, ps_a_ori[:, 19:69])
            audios['recon_a'].append(result_a['audio_gen'][0].cpu().numpy())

            result_b = networks['Synthesis'](lps_b, s_b, e_b, ps_b_ori[:, 19:69])
            audios['recon_b'].append(result_b['audio_gen'][0].cpu().numpy())

        # vc
        s_a = torch.mean(torch.cat(logs['s_a'], dim=0), dim=0, keepdim=True)
        s_b = torch.mean(torch.cat(logs['s_b'], dim=0), dim=0, keepdim=True)

        sound_a = wav_to_Sound(audios['gt_a'][idx].copy(), 22050)
        pitch_a = parselmouth.praat.call(sound_a, "To Pitch", 0, 75, 600)
        pitch_median_a_temp = parselmouth.praat.call(pitch_a, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        if not math.isnan(pitch_median_a_temp):
            pitch_median_a = pitch_median_a_temp

        sound_b = wav_to_Sound(audios['gt_b'][idx].copy(), 22050)
        pitch_b = parselmouth.praat.call(sound_b, "To Pitch", 0, 75, 600)
        pitch_median_b_temp = parselmouth.praat.call(pitch_b, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
        if not math.isnan(pitch_median_b_temp):
            pitch_median_b = pitch_median_b_temp

        midi_diff = hz_diff_to_midi_diff(pitch_median_a, pitch_median_b)
        print('pitch', pitch_median_a, pitch_median_b, 'midi', midi_diff)
        midi_diff = int(midi_diff)
        if midi_diff > 11:
            midi_diff = 11
        if midi_diff < -11:
            midi_diff = -11

        with torch.no_grad():
            result_a2b = networks['Synthesis'](lps_a, s_b, e_a, ps_a_ori[:, 19 + midi_diff:69 + midi_diff])

            audios['text_a_spk_b'].append(result_a2b['audio_gen'][0].cpu().numpy())

            result_b2a = networks['Synthesis'](lps_b, s_a, e_b, ps_b_ori[:, 19 - midi_diff:69 - midi_diff])
            audios['text_b_spk_a'].append(result_b2a['audio_gen'][0].cpu().numpy())

        # test time self adaptation
        for i in range(args.tsa_loop):
            lps_tsa = lps_a.detach().clone()
            lps_tsa = tsa_helper(lps_tsa)
            result = networks['Synthesis'](lps_tsa, s_a.clone().detach(), e_a, ps_a_ori[:, 19:69])

            opt.zero_grad()
            a_mel_22k = logs['a_mel_22k'][idx]
            loss_mel = torch.nn.functional.l1_loss(a_mel_22k, result['gen_mel'])
            loss_mel.backward()
            opt.step()

        with torch.no_grad():
            lps_a = tsa_helper(lps_a)
            result = networks['Synthesis'](lps_a, s_a, e_a, ps_a_ori[:, 19:69])
            audios['recon_a_tsa'].append(result['audio_gen'][0].cpu().numpy())

            result = networks['Synthesis'](lps_a, s_b, e_a, ps_a_ori[:, 19 + midi_diff:69 + midi_diff])
            audios['text_a_spk_b_tsa'].append(result['audio_gen'][0].cpu().numpy())

    final_audios = {}
    for key, value in audios.items():
        try:
            final_audios[key] = np.concatenate(value, axis=-1)
            print(key, len(value), final_audios[key].shape)
        except Exception as e:
            print(key, e)

    # end region

    # region tsm
    cats = {key: torch.cat(logs[key], dim=-1) for key in logs.keys()}
    to_interpolate = {
        'lps_a': cats['lps_a'], 'lps_b': cats['lps_b'],
        'e_a': cats['e_a'], 'e_b': cats['e_b'],
        'ps_a': cats['ps_a'], 'ps_b': cats['ps_b']}

    scale = 2
    interpolated_cats = {key: torch.nn.functional.interpolate(value, scale_factor=scale, mode='linear')
                         for key, value in to_interpolate.items()}

    audios.update({
        'tsm_a': [],
        'tsm_b': [],
    })
    for idx in range(interpolated_cats['lps_a'].shape[-1] // 74):
        lps_a = interpolated_cats['lps_a'][..., 74 * idx:74 * (idx + 1)]
        lps_b = interpolated_cats['lps_b'][..., 74 * idx:74 * (idx + 1)]
        s_a = torch.mean(torch.cat(logs['s_a'], dim=0), dim=0, keepdim=True)
        s_b = torch.mean(torch.cat(logs['s_b'], dim=0), dim=0, keepdim=True)
        e_a = interpolated_cats['e_a'][..., 128 * idx:128 * (idx + 1)]
        e_b = interpolated_cats['e_b'][..., 128 * idx:128 * (idx + 1)]
        ps_a_ori = interpolated_cats['ps_a'][..., 128 * idx:128 * (idx + 1)]
        ps_b_ori = interpolated_cats['ps_b'][..., 128 * idx:128 * (idx + 1)]

        # recon
        with torch.no_grad():
            result_a = networks['Synthesis'](lps_a, s_a, e_a, ps_a_ori[:, 19:69])
            audios['tsm_a'].append(result_a['audio_gen'][0].cpu().numpy())

            result_b = networks['Synthesis'](lps_b, s_b, e_b, ps_b_ori[:, 19:69])
            audios['tsm_b'].append(result_b['audio_gen'][0].cpu().numpy())

    # endregion

    return final_audios


if __name__ == '__main__':
    import os
    from scipy.io import wavfile

    final_audios = main()

    dir_save = './temp_result'
    os.makedirs(dir_save, exist_ok=True)
    for key, value in final_audios.items():
        sample_rate = 22050
        try:
            wavfile.write(
                os.path.join(dir_save, f'{key}.wav'),
                sample_rate,
                value.squeeze().astype(np.float32))
        except Exception as e:
            print(key, e)
