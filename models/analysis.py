import math

import numpy as np
import torch
import transformers

from models.ecapa import ECAPA_TDNN
from models.yin import *


class Linguistic(torch.nn.Module):
    def __init__(self, conf=None):
        super(Linguistic, self).__init__()
        self.conf = conf

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()

    def forward(self, x):
        # x.shape: B x t
        with torch.no_grad():
            outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[1]
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # for module in self.children():
        #     module.train(mode)
        return self


class Speaker(torch.nn.Module):
    def __init__(self, conf=None):
        super(Speaker, self).__init__()
        self.conf = conf

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()

        # c_in = 1024 for wav2vec2
        # original paper used 512 and 192 for c_mid and c_out, respectively
        self.spk = ECAPA_TDNN(c_in=1024, c_mid=512, c_out=192)

    def forward(self, x):
        # x.shape: B x t
        with torch.no_grad():
            outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[12]
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        y = self.spk(y)
        return y

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # for module in self.children():
        #     module.train(mode)
        self.spk.train(mode)
        return self


class Energy(torch.nn.Module):
    def __init__(self, conf=None):
        super(Energy, self).__init__()
        self.conf = conf

    def forward(self, mel):
        # mel: B x t x C
        # For the energy feature, we simply took an average from a log-mel spectrogram along the frequency axis.
        y = torch.mean(mel, dim=1, keepdim=True)  # B x 1(channel) x t
        return y


class Pitch(torch.nn.Module):
    def __init__(self, conf=None):
        super(Pitch, self).__init__()
        self.conf = conf

    @staticmethod
    def midi_to_hz(m, semitone_range=12):
        f = 440 * math.pow(2, (m - 69) / semitone_range)
        return f

    @staticmethod
    def midi_to_lag(m, sr, semitone_range=12):
        f = Pitch.midi_to_hz(m, semitone_range)
        lag = sr / f
        return lag

    @staticmethod
    def yingram_from_cmndf(cmndfs: torch.Tensor, ms: list, sr=22050) -> torch.Tensor:
        c_ms = np.asarray([Pitch.midi_to_lag(m, sr) for m in ms])
        c_ms = torch.from_numpy(c_ms).to(cmndfs.device)
        c_ms_ceil = torch.ceil(c_ms).long().to(cmndfs.device)
        c_ms_floor = torch.floor(c_ms).long().to(cmndfs.device)

        y = (cmndfs[:, c_ms_ceil] - cmndfs[:, c_ms_floor]) / (c_ms_ceil - c_ms_floor).unsqueeze(0) * (
                c_ms - c_ms_floor).unsqueeze(0) + cmndfs[:, c_ms_floor]
        return y

    @staticmethod
    def yingram(x: torch.Tensor, W=2048, tau_max=2048, sr=22050, w_step=256):
        # x.shape: t
        w_len = W

        startFrames = list(range(0, x.shape[-1] - w_len, w_step))
        startFrames = np.asarray(startFrames)
        # times = startFrames / sr
        frames = [x[..., t:t + W] for t in startFrames]
        frames_torch = torch.stack(frames, dim=0).to(x.device)
        # frames = np.asarray(frames)
        # frames_torch = torch.from_numpy(frames).to(x.device)
        dfs = differenceFunctionTorch(frames_torch, frames_torch.shape[-1], tau_max)
        cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, tau_max)
        yingram = Pitch.yingram_from_cmndf(cmndfs, list(range(5, 85)), sr)
        return yingram

    @staticmethod
    def yingram_batch(x, W=2048, tau_max=2048, sr=22050, w_step=256):
        """

        params:
            x.shape: B, n
        """
        batch_results = []
        for i in range(len(x)):
            yingram = Pitch.yingram(x[i], W, tau_max, sr, w_step)
            batch_results.append(yingram)
        result = torch.stack(batch_results, dim=0).float()
        result = result.permute((0, 2, 1)).to(x.device)
        return result


class Analysis(torch.nn.Module):
    def __init__(self, conf):
        super(Analysis, self).__init__()
        self.conf = conf

        self.linguistic = Linguistic()
        self.speaker = Speaker()
        self.energy = Energy()
        self.pitch = Pitch()


if __name__ == '__main__':
    import torch

    wav = torch.randn(2, 20000)
    mel = torch.randn(2, 80, 128)

    linguistic = Linguistic()
    speaker = Speaker()
    energy = Energy()
    pitch = Pitch()

    with torch.no_grad():
        lps = linguistic(wav)
        print(lps.shape)

        s = speaker(wav)
        print(s.shape)

        e = energy(mel)
        print(e.shape)

        ps = pitch(wav)
        print(ps.shape)
