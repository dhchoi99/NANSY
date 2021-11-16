import importlib

hifigan = importlib.import_module('models.hifi-gan')
hifigan_vocoder = getattr(hifigan, 'Generator')
import json
import os

import torch
from torch import nn
import torch.nn.functional as F


class ConditionalLayerNorm(nn.Module):
    def __init__(self, embedding_dim, normalize_embedding=True):
        super(ConditionalLayerNorm, self).__init__()
        self.normalize_embedding = normalize_embedding

        self.linear_scale = nn.Linear(embedding_dim, 1)
        self.linear_bias = nn.Linear(embedding_dim, 1)

    def forward(self, x, embedding):
        if self.normalize_embedding:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        scale = self.linear_scale(embedding).unsqueeze(-1)  # shape: (B, 1, 1)
        bias = self.linear_bias(embedding).unsqueeze(-1)  # shape: (B, 1, 1)

        # out = self.norm(x)
        out = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.var(x, dim=-1, keepdim=True)
        out = scale * out + bias
        return out


class ConvGLU(nn.Module):
    def __init__(self, channel, ks, dilation, embedding_dim=192, use_cLN=False):
        super(ConvGLU, self).__init__()

        self.dropout = nn.Dropout()
        self.conv = nn.Conv1d(channel, channel * 2, kernel_size=ks, stride=1, padding=(ks - 1) // 2 * dilation,
                              dilation=dilation)
        self.glu = nn.GLU(dim=1)  # channel-wise

        self.use_cLN = use_cLN
        if self.use_cLN:
            self.norm = ConditionalLayerNorm(embedding_dim)

    def forward(self, x, speaker_embedding=None):
        y = self.dropout(x)
        y = self.conv(y)
        y = self.glu(y)
        y = y + x

        if self.use_cLN and speaker_embedding is not None:
            y = self.norm(y, speaker_embedding)
        return y


class PreConv(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super(PreConv, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=1, dilation=1),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv1d(c_mid, c_mid, kernel_size=1, dilation=1),
            nn.LeakyReLU(),
            nn.Dropout(),

            nn.Conv1d(c_mid, c_out, kernel_size=1, dilation=1),
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Generator(nn.Module):
    def __init__(self, c_in=1024, c_preconv=512, c_mid=128, c_out=80):
        super(Generator, self).__init__()

        self.network1 = nn.Sequential(
            PreConv(c_in, c_preconv, c_mid),

            ConvGLU(c_mid, ks=3, dilation=1, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=3, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=9, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=27, use_cLN=False),

            ConvGLU(c_mid, ks=3, dilation=1, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=3, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=9, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=27, use_cLN=False),

            ConvGLU(c_mid, ks=3, dilation=1, use_cLN=False),
            ConvGLU(c_mid, ks=3, dilation=3, use_cLN=False),
        )

        self.LR = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(c_mid + 1, c_mid + 1, kernel_size=1, stride=1))

        self.network3 = nn.ModuleList([
            ConvGLU(c_mid + 1, ks=3, dilation=1, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=3, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=9, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=27, use_cLN=True),

            ConvGLU(c_mid + 1, ks=3, dilation=1, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=3, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=9, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=27, use_cLN=True),

            ConvGLU(c_mid + 1, ks=3, dilation=1, use_cLN=True),
            ConvGLU(c_mid + 1, ks=3, dilation=3, use_cLN=True),
        ])

        self.lastConv = nn.Conv1d(c_mid + 1, c_out, kernel_size=1, dilation=1)

    def forward(self, x, energy, speaker_embedding):
        # x.shape: B x C x t
        # energy.shape: B x 1 x t
        # embedding.shape: B x d x 1
        y = self.network1(x)
        B, C, _ = y.shape

        y = F.interpolate(y, energy.shape[-1])  # B x C x d
        y = torch.cat((y, energy), dim=1)  # channel-wise concat
        y = self.LR(y)

        for module in self.network3:  # doing this since sequential takes only 1 input
            y = module(y, speaker_embedding)
        y = self.lastConv(y)
        return y


#####

class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid=128, c_out=128):
        super(ResBlock, self).__init__()
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=1, dilation=3)

        self.leaky_relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(c_mid, c_out, kernel_size=3, stride=1, padding=1, dilation=3)

        self.conv3 = nn.Conv1d(c_in, c_out, kernel_size=1, dilation=1)

    def forward(self, x):
        y = self.conv1(self.leaky_relu1(x))
        y = self.conv2(self.leaky_relu2(y))
        y = y + self.conv3(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, c_in=80, c_mid=128, c_out=192):
        super(Discriminator, self).__init__()

        self.network1 = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=3, stride=1, padding=1, dilation=1),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
            ResBlock(c_mid, c_mid, c_mid),
        )

        self.res = ResBlock(c_mid, c_mid, c_out)
        self.conv = nn.Conv1d(c_mid, 1, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x, speaker_embedding):
        y = self.network1(x)

        y1 = self.conv(y)
        y2 = self.res(y)
        y2 = torch.cat((y2, speaker_embedding), dim=-1)  # TODO tile speaker embedding

        z = y1 + y2
        return z


#####
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Vocoder(nn.Module):
    def __init__(self, path_config='./configs/hifi-gan/config.json', path_ckpt='./configs/hifi-gan/generator_v1'):
        super(Vocoder, self).__init__()

        with open(path_config) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        self.network = hifigan_vocoder(h)

        state_dict_g = self.load_checkpoint(path_ckpt)
        self.network.load_state_dict(state_dict_g['generator'])

        for param in self.network.parameters():
            param.requires_grad = False

    @staticmethod
    def load_checkpoint(filepath):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath)
        print("Complete.")
        return checkpoint_dict


if __name__ == '__main__':
    lps = torch.randn(2, 1024, 128)
    s = torch.randn(2, 192)
    e = torch.randn(2, 128)
    ps = torch.randn(2, 80, 128)

    g1 = Generator(1024, 512, 128, 80)
    y1 = g1(lps, e, s)
    print(y1.shape)

    g2 = Generator(80, 512, 128, 80)
    y2 = g2(ps, e, s)
    print(y2.shape)
