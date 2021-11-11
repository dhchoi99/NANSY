import math

import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


def mel_from_path(args_audio, wavpath, noise_factor: float = 0):
    raw_wav = load_wav(wavpath, args_audio.sample_rate)
    noise = np.random.randn(len(raw_wav))
    wav = raw_wav + noise_factor * noise
    wav = wav.astype(type(raw_wav[0]))
    mel = melspectrogram(args_audio, wav).T
    return mel


def mel_from_audio(args_audio, raw_wav, noise_factor: float = 0):
    noise = np.random.randn(len(raw_wav))
    wav = raw_wav + noise_factor * noise
    wav = wav.astype(type(raw_wav[0]))
    mel = melspectrogram(args_audio, wav).T
    return mel


def crop_audio_window(args_audio, spec, current_frame: int, fps=25, pad_blank=False):
    # num_Frames = (T x hop_size * fps) / sample_rate
    mel_frame_ratio = args_audio.sample_rate / args_audio.hop_size
    mel_window = math.ceil(mel_frame_ratio * args_audio.frame_window / float(fps))
    current_idx = int(mel_frame_ratio * current_frame / float(fps))
    start_idx = current_idx - mel_window // 2
    end_idx = start_idx + mel_window
    if start_idx >= 0 and end_idx < len(spec):
        mel = spec[start_idx:end_idx]
    else:
        if pad_blank:
            blank_wav = np.random.randn(50000) * 0
            blank_mel = melspectrogram(args_audio, blank_wav).T

            mels = []
            if start_idx < 0 and end_idx < len(spec):
                for _ in range(-start_idx):
                    mels.append(blank_mel[0])
                for i in range(0, end_idx):
                    mels.append(spec[i])
            elif start_idx >= 0 and end_idx >= len(spec):
                for i in range(start_idx, len(spec)):
                    mels.append(spec[i])
                for _ in range(end_idx - len(spec)):
                    mels.append(blank_mel[0])
            elif start_idx > len(spec):
                raise ValueError
            else:
                raise NotImplementedError
            mel = np.stack(mels, axis=0)
        else:
            raise NotImplementedError

    assert len(mel) == mel_window, f'{len(spec)}_{len(mel)}_{mel_window}_{current_frame}_{current_idx}_{start_idx}'
    return mel


##################################

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size(opt):
    hop_size = opt.hop_size
    if hop_size is None:
        assert opt.frame_shift_ms is not None
        hop_size = int(opt.frame_shift_ms / 1000 * opt.sample_rate)
    return hop_size


def linearspectrogram(opt, wav):
    D = _stft(opt, preemphasis(wav, opt.preemphasis, opt.preemphasize))
    S = _amp_to_db(opt, np.abs(D)) - opt.ref_level_db

    if opt.signal_normalization:
        return _normalize(opt, S)
    return S


def melspectrogram(opt, wav):
    D = _stft(opt, preemphasis(wav, opt.preemphasis, opt.preemphasize))
    S = _amp_to_db(opt, _linear_to_mel(opt, np.abs(D))) - opt.ref_level_db

    if opt.signal_normalization:
        return _normalize(opt, S)
    return S


def _lws_processor(opt):
    import lws
    return lws.lws(opt.n_fft, get_hop_size(opt), fftsize=opt.win_size, mode="speech")


def _stft(opt, y):
    if opt.use_lws:
        return _lws_processor(opt).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=opt.n_fft, hop_length=get_hop_size(opt), win_length=opt.win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


# Conversions
_mel_basis = None


def _linear_to_mel(opt, spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(opt)
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis(opt):
    assert opt.fmax <= opt.sample_rate // 2
    return librosa.filters.mel(opt.sample_rate, opt.n_fft, n_mels=opt.num_mels,
                               fmin=opt.fmin, fmax=opt.fmax)


def _amp_to_db(opt, x):
    min_level = np.exp(opt.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(opt, S):
    if opt.allow_clipping_in_normalization:
        if opt.symmetric_mels:
            return np.clip((2 * opt.max_abs_value) * ((S - opt.min_level_db) / (-opt.min_level_db)) - opt.max_abs_value,
                           -opt.max_abs_value, opt.max_abs_value)
        else:
            return np.clip(opt.max_abs_value * ((S - opt.min_level_db) / (-opt.min_level_db)), 0, opt.max_abs_value)

    assert S.max() <= 0 and S.min() - opt.min_level_db >= 0
    if opt.symmetric_mels:
        return (2 * opt.max_abs_value) * ((S - opt.min_level_db) / (-opt.min_level_db)) - opt.max_abs_value
    else:
        return opt.max_abs_value * ((S - opt.min_level_db) / (-opt.min_level_db))


def _denormalize(opt, D):
    if opt.allow_clipping_in_normalization:
        if opt.symmetric_mels:
            return (((np.clip(D, -opt.max_abs_value,
                              opt.max_abs_value) + opt.max_abs_value) * -opt.min_level_db / (2 * opt.max_abs_value))
                    + opt.min_level_db)
        else:
            return ((np.clip(D, 0, opt.max_abs_value) * - opt.min_level_db / opt.max_abs_value) + opt.min_level_db)

    if opt.symmetric_mels:
        return (((D + opt.max_abs_value) * -opt.min_level_db / (2 * opt.max_abs_value)) + opt.min_level_db)
    else:
        return ((D * -opt.min_level_db / opt.max_abs_value) + opt.min_level_db)


def read_wav_from_bytes(opt, bytes_wrapped, to_sr=16000):
    from_sr, wav = wavfile.read(bytes_wrapped)

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    from_sr = float(from_sr)
    if from_sr != to_sr:
        from_secs = len(wav) / from_sr
        to_secs = int(from_secs * to_sr)
        wav = signal.resample(wav, to_secs)
        wav = wav.astype(np.int16)

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return wav
