import argparse

import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch
import torchaudio.functional as AF
from tqdm.notebook import tqdm, trange

from datasets.custom import CustomDataset
from models.analysis import Analysis
from models.synthesis import Synthesis, Discriminator
from inference import pl_checkpoint_to_torch_checkpoints
import utils.mel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_audio_conf', type=str, default='configs/audio/22k.yaml',
                        help='')
    parser.add_argument('--path_ckpt', type=str, required=True,
                        help='path to pl checkpoint')
    parser.add_argument('--path_audio_gt', type=str, required=True,
                        help='path to audio(22k) with same speaker')
    parser.add_argument('--path_audio_gen', type=str, required=True,
                        help='path to generated audio(22k)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    conf_audio = OmegaConf.load(args.path_audio_conf)
    conf = DictConfig({'audio': conf_audio})

    self = CustomDataset(conf)

    wav_22k_numpy, wav_22k_torch = self.load_wav(args.path_audio_gt, 22050)
    wav_16k_torch = AF.resample(wav_22k_torch, 22050, 16000)
    mel_22k_torch = self.load_mel(args.path_audio_gt, sr=22050)

    data_ckpt = torch.load(args.path_ckpt, map_location='cpu')
    state_dict = data_ckpt['state_dict']
    new_state_dict = pl_checkpoint_to_torch_checkpoints(state_dict)

    analysis = Analysis(None)
    analysis.load_state_dict(new_state_dict['Analysis'])
    analysis.to(args.device).eval()

    synthesis = Synthesis(None)
    synthesis.load_state_dict(new_state_dict['Synthesis'])
    synthesis.to(args.device).eval()

    discriminator = Discriminator(None)
    discriminator.load_state_dict(new_state_dict['Discriminator'])
    discriminator.to(args.device).eval()

    networks = {'Analysis': analysis, 'Synthesis': synthesis, 'Discriminator': discriminator}

    preds = []
    for idx in trange(0, mel_22k_torch.shape[-1], self.mel_window):
        return_data = {}
        mel_start = idx
        pos_time_idxs = self.get_time_idxs(mel_start)

        mel_22k = self.crop_audio(mel_22k_torch, pos_time_idxs[0], pos_time_idxs[1],
                                  padding_value=self.mel_padding_value)
        return_data['gt_mel_22k'] = mel_22k
        return_data['gt_log_mel_22k'] = mel_22k

        wav_16k = self.crop_audio(wav_16k_torch, pos_time_idxs[3], pos_time_idxs[5])
        # return_data['gt_audio_16k_f'] = f(wav_16k, sr=16000)
        return_data['gt_audio_16k_f'] = wav_16k
        return_data['gt_audio_16k'] = wav_16k

        wav_22k = self.crop_audio(wav_22k_torch, pos_time_idxs[4], pos_time_idxs[6])
        wav_22k_yin = self.crop_audio(wav_22k_torch, pos_time_idxs[4], pos_time_idxs[7])
        return_data['gt_audio_22k'] = wav_22k
        # return_data['gt_audio_22k_g'] = g(wav_22k_yin, sr=22050)
        return_data['gt_audio_22k_g'] = wav_22k_yin

        batch = {key: value.unsqueeze(0).to(args.device) for key, value in return_data.items()}

        with torch.no_grad():
            lps = networks['Analysis'].linguistic(batch['gt_audio_16k_f'])
            s_src = networks['Analysis'].speaker(batch['gt_audio_16k_f'])
            e = networks['Analysis'].energy(batch['gt_mel_22k'])
            ps = networks['Analysis'].pitch.yingram_batch(batch['gt_audio_22k_g'])
            ps = ps[:, 19:69]

        with torch.no_grad():
            result = networks['Synthesis'](lps, s_src, e, ps)

        mel = utils.mel.mel_spectrogram(
            result['audio_gen'][0], 1024, 80, 22050, 256, 1024, 0, 8000
        )
        with torch.no_grad():
            s_gen = networks['Analysis'].speaker(result['audio_gen'][0])

        with torch.no_grad():
            # pred = networks['Discriminator'](batch['gt_mel_22k'], s_src, s_gen)
            pred = networks['Discriminator'](mel, s_gen, s_src)
            preds.append(pred)

    preds = [torch.mean(torch.sigmoid(pred)).item() for pred in preds]
    print(preds)
    print(torch.mean(torch.Tensor(preds)))

if __name__ == '__main__':
    main()
