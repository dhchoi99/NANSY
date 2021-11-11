import torch
import transformers

from models.ecapa import ECAPA_TDNN


class Linguistic(torch.nn.Module):
    def __init__(self, conf=None):
        super(Linguistic, self).__init__()
        self.conf = conf

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[1]
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y


class Speaker(torch.nn.Module):
    def __init__(self, conf=None):
        super(Speaker, self).__init__()
        self.conf = conf

        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # c_in = 1024 for wav2vec2
        # original paper used 512 and 192 for c_mid and c_out, respectively
        self.spk = ECAPA_TDNN(c_in=1024, c_mid=512, c_out=192)

    def forward(self, x):
        outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[12]
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        y = self.spk(y)
        return y


class Energy(torch.nn.Module):
    def __init__(self, conf=None):
        super(Energy, self).__init__()
        self.conf = conf

    def forward(self, mel):
        # mel: B x t x C
        # For the energy feature, we simply took an average from a log-mel spectrogram along the frequency axis.
        y = torch.mean(mel, dim=-1)
        return y


class Pitch(torch.nn.Module):
    def __init__(self, conf=None):
        super(Pitch, self).__init__()
        self.conf = conf

    def forward(self, x):
        raise NotImplementedError
