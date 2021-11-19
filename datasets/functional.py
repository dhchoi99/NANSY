# https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887
import math
import random

import librosa
import numpy as np
import parselmouth
import torch
import torchaudio.functional as AF


# __init__(self: parselmouth.Sound, other: parselmouth.Sound) -> None \
# __init__(self: parselmouth.Sound, values: numpy.ndarray[numpy.float64], sampling_frequency: Positive[float] = 44100.0, start_time: float = 0.0) -> None \
# __init__(self: parselmouth.Sound, file_path: str) -> None
def wav_to_Sound(wav, sampling_frequency=22050):
    if isinstance(wav, parselmouth.Sound):
        sound = wav
    elif isinstance(wav, np.ndarray):
        sound = parselmouth.Sound(wav, sampling_frequency=sampling_frequency)
    elif isinstance(wav, list):
        wav_np = np.asarray(wav)
        sound = parselmouth.Sound(np.asarray(wav_np), sampling_frequency=sampling_frequency)
    else:
        raise NotImplementedError
    return sound


def wav_to_Tensor(wav):
    if isinstance(wav, np.ndarray):
        wav_tensor = torch.from_numpy(wav)
    elif isinstance(wav, torch.Tensor):
        wav_tensor = wav
    elif isinstance(wav, parselmouth.Sound):
        wav_np = wav.values
        wav_tensor = torch.from_numpy(wav_np)
    else:
        raise NotImplementedError
    return wav_tensor


def apply_formant_and_pitch_shift(sound: parselmouth.Sound, formant_shift_ratio=1., pitch_shift_ratio=1.,
                                  pitch_range_ratio=1., duration_factor=1.):
    # pitch = sound.to_pitch()
    if pitch_shift_ratio != 1.:
        try:
            pitch = parselmouth.praat.call(sound, "To Pitch", 0, 75, 600)
            # pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
            try:
                pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
                if not math.isnan(pitch_median):
                    # print('not nan')
                    new_pitch_median = pitch_median * pitch_shift_ratio
                    if math.isnan(new_pitch_median):
                        new_pitch_median = 0.0
                else:
                    # print('nan')
                    new_pitch_median = 0.0
            except:
                print('e2')
                new_pitch_median = 0.0
        except:
            print('e1')
            new_pitch_median = 0.0

        try:
            new_sound = parselmouth.praat.call(
                sound, "Change gender", 75, 600,
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor
            )
        except Exception as e:
            print('e3')
            print(e)
            print(formant_shift_ratio)
            print(new_pitch_median)
            print(pitch_range_ratio)
            print(duration_factor)
            new_sound = parselmouth.praat.call(
                sound, "Change gender", 75, 600,
                formant_shift_ratio,
                0.0,
                pitch_range_ratio,
                duration_factor
            )
    else:
        new_sound = parselmouth.praat.call(
            sound, "Change gender", 75, 600,
            formant_shift_ratio,
            0.0,
            pitch_range_ratio,
            duration_factor
        )
    return new_sound


# fs & pr
def formant_and_pitch_shift(sound):
    formant_shifting_ratio = random.uniform(1, 1.4)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        formant_shifting_ratio = 1 / formant_shifting_ratio

    pitch_shift_ratio = random.uniform(1, 2)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        pitch_shift_ratio = 1 / pitch_shift_ratio

    pitch_range_ratio = random.uniform(1, 1.5)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        pitch_range_ratio = 1 / pitch_range_ratio

    sound_new = apply_formant_and_pitch_shift(
        sound,
        formant_shift_ratio=formant_shifting_ratio,
        pitch_shift_ratio=pitch_shift_ratio,
        pitch_range_ratio=pitch_range_ratio,
        duration_factor=1.
    )
    return sound_new


# fs
def formant_shift(sound):
    formant_shifting_ratio = random.uniform(1, 1.4)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        formant_shifting_ratio = 1 / formant_shifting_ratio

    sound_new = apply_formant_and_pitch_shift(
        sound,
        formant_shift_ratio=formant_shifting_ratio,
    )
    return sound_new


def power_ratio(r: float, a: float, b: float):
    return a * math.pow((b / a), r)


# peq
def parametric_equalizer(wav: torch.Tensor, sr) -> torch.Tensor:
    cutoff_low_freq = 60.
    cutoff_high_freq = 10000.

    q_min = 2
    q_max = 5

    num_filters = 8 + 2  # 8 for peak, 2 for high/low
    key_freqs = [
        power_ratio(float(z) / num_filters, cutoff_low_freq, cutoff_high_freq)
        for z in range(num_filters)
    ]
    gains = [random.uniform(-12, 12) for _ in range(num_filters)]
    Qs = [
        power_ratio(random.uniform(0, 1), q_min, q_max)
        for _ in range(num_filters)
    ]

    # peak filters
    for i in range(1, 9):
        wav = apply_iir_filter(
            wav,
            ftype='peak',
            dBgain=gains[i],
            cutoff_freq=key_freqs[i],
            sample_rate=sr,
            Q=Qs[i]
        )

    # high-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='high',
        dBgain=gains[-1],
        cutoff_freq=key_freqs[-1],
        sample_rate=sr,
        Q=Qs[-1]
    )

    # low-shelving filter
    wav = apply_iir_filter(
        wav,
        ftype='low',
        dBgain=gains[0],
        cutoff_freq=key_freqs[0],
        sample_rate=sr,
        Q=Qs[0]
    )

    return wav


# https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
# implemented using the cookbook
def lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
    a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

    a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
    a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
    return b0, b1, b2, a0, a1, a2


def peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q):
    A = math.pow(10, dBgain / 40.)

    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A

    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return b0, b1, b2, a0, a1, a2


def apply_iir_filter(wav: torch.Tensor, ftype, dBgain, cutoff_freq, sample_rate, Q):
    if ftype == 'low':
        b0, b1, b2, a0, a1, a2 = lowShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'high':
        b0, b1, b2, a0, a1, a2 = highShelf_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    elif ftype == 'peak':
        b0, b1, b2, a0, a1, a2 = peaking_coeffs(dBgain, cutoff_freq, sample_rate, Q)
    else:
        raise NotImplementedError
    return_wav = AF.biquad(wav, b0, b1, b2, a0, a1, a2)
    return return_wav


peq = parametric_equalizer


# def f(wav: torch.Tensor, sr: int) -> torch.Tensor:
#     """
#
#     :param wav: torch.Tensor of shape (N,)
#     :param sr: sampling rate
#     :return: torch.Tensor of shape (M, )
#     """
#     wav = peq(wav, sr)
#     wav_numpy = wav.numpy()
#     sound = wav_to_Sound(wav_numpy, sampling_frequency=sr)
#     sound = formant_and_pitch_shift(sound)
#     return torch.from_numpy(sound.values)


def g(wav: torch.Tensor, sr: int) -> torch.Tensor:
    wav = peq(wav, sr)
    wav_numpy = wav.numpy()
    sound = wav_to_Sound(wav_numpy, sampling_frequency=sr)
    sound = formant_shift(sound)
    wav = torch.from_numpy(sound.values).float()
    return wav


def f(wav: torch.Tensor, sr: int) -> torch.Tensor:
    wav = peq(wav, sr)
    wav_numpy = wav.numpy()
    sound = wav_to_Sound(wav_numpy, sampling_frequency=sr)
    sound = formant_shift(sound)
    wav_numpy = sound.values

    n_steps = random.uniform(-24, 24)
    wav_numpy = librosa.effects.pitch_shift(
        wav_numpy[0], sr=sr,
        n_steps=n_steps, bins_per_octave=12
    )
    wav = torch.from_numpy(wav_numpy).float().unsqueeze(0)
    return wav
