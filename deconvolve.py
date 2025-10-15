# coding:utf-8
"""
    :module: deconvolve.py
    :description: Deconvolve / process impulse responses functions
    :author: Michel 'Mitch' Pecqueur
    :date: 2024.04
"""

import math

import numpy as np
from numpy.fft import fft, ifft


def deconvolve(audio: np.ndarray, reference: np.ndarray, lambd: float = 1e-3, mode: str = 'minmax_sum') -> np.ndarray:
    """
    Deconvolve audio from a reference sound (typically a sweep) to an impulse response

    :param audio: Convolved audio
    :param reference: Reference audio
    :param lambd: Peak signal-to-noise ratio
    :param mode: Match length mode between audio and ref, "min", "max" or 'minmax_sum' (to alleviate wrap-around)

    :return: Resulting IR
    """
    au_nch, ref_nch = audio.ndim, reference.ndim

    result = None
    for i in range(au_nch):
        if au_nch > 1:
            conv_data = audio[:, i]
        else:
            conv_data = audio

        if ref_nch > 1:
            ref_data = reference[:, i]
        else:
            ref_data = reference

        # Match convolved and reference audio length
        r_l, c_l = len(ref_data), len(conv_data)
        mn_l, mx_l = min(r_l, c_l), max(r_l, c_l)
        if mode == 'min':
            length = mn_l
            kernel = ref_data[:length]
            conv_data = conv_data[:length]
        else:
            length = mx_l
            pad_length = (length, length + mn_l)[mode == 'minmax_sum']
            conv_data = np.pad(conv_data, (0, max(0, pad_length - c_l)), mode='constant', constant_values=0)
            kernel = np.pad(ref_data, (0, max(0, pad_length - r_l)), mode='constant', constant_values=0)

        # Wiener Deconvolution
        # Taken fom "Example of Wiener deconvolution in Python"
        # Written 2015 by Dan Stowell. Public domain.
        # https://gist.github.com/danstowell/f2d81a897df9e23cc1da

        fft_k = fft(kernel)
        deconv = np.real(ifft(fft(conv_data) * np.conj(fft_k) / (fft_k * np.conj(fft_k) + lambd ** 2)))[:length]

        if result is None:
            result = deconv
        else:
            result = np.column_stack((result, deconv))

    return result


def compensate_ir(audio: np.ndarray, mode: str = 'rms', sr: int = 48000) -> np.ndarray:
    """
    Compensate impulse response volume so convolved audio keeps approximately the same gain as original
    :param audio: Input impulse response
    :param mode: Normalization mode, 'peak' or 'rms'
    :param sr: Sampling rate
    :return: processed IR
    """
    nch = audio.ndim
    length = len(audio)

    vol_func = {'peak': peak, 'rms': rms}[mode]

    test_tone = generate_sweep(length / sr, sr, db=-6, start_freq=20, window=True, os=1)

    values = []
    for c in range(nch):
        if nch > 1:
            data = audio[:, c]
        else:
            data = audio
        orig_vol = vol_func(test_tone)
        conv = np_fftconvolve(test_tone, data, mode='full')[:length]
        conv_vol = vol_func(conv)
        factor = orig_vol / conv_vol
        values.append(factor)
    gain = np.mean(values)

    return audio * gain


def generate_sweep(duration: float = 4, sr: int = 48000, db: float = -6, start_freq: float = 20, window: bool = True,
                   os: int = 1):
    """
    Generate logarithmic sweep tone
    :param duration: in seconds
    :param sr: Sample Rate
    :param db: Volume
    :param start_freq: in Hz
    :param window: Apply Tukey window
    :param os: Oversampling factor
    :return: Generated audio
    """
    length = int(duration * sr)
    end_freq = sr / 2

    pad = np.array([0])
    if window:
        pad = freq_to_period(start_freq, sr) * np.array([1, 1], dtype=np.int32)

    freq = np.logspace(np.log10(start_freq), np.log10(end_freq), (length - sum(pad)) * os, endpoint=True)

    freq[[0, -1]] = [start_freq, end_freq]
    freq = np.pad(freq, pad_width=pad * os, mode='edge')

    phase = np.cumsum(2 * np.pi * freq / (sr * os))
    phase -= phase[0]  # Start from 0
    sweep = np.sin(phase) * db_to_lin(db)

    if window:
        w = sweep_window(length * os, int(freq_to_period(start_freq, sr * os)))
        sweep *= w

    if os > 1:
        sweep = np_decimate(sweep, os, n=None)

    return sweep


def generate_impulse(duration: float = 4, sr: int = 48000, db: float = -0.5) -> np.ndarray:
    """
    Generate an impulse click
    """
    length = int(duration * sr)
    impulse = np.zeros(length)
    impulse[1] = db_to_lin(db)
    return impulse


def convolve(audio: np.ndarray, ir: np.ndarray, comp_ir: bool = True, wet: float = 1.0, sr: int = 48000) -> np.ndarray:
    """
    Convolve input audio with an impulse response
    :param audio: Input audio
    :param ir: Impulse response
    :param comp_ir: Compensate IR gain
    :param wet: Blend between processed and original audio
    :param sr: Sample Rate
    :return: Processed audio
    """
    au_nch, ir_nch = audio.ndim, ir.ndim
    au_l, ir_l = len(audio), len(ir)

    if comp_ir:
        ir = compensate_ir(ir, mode='rms', sr=sr)

    # Match number of channels between IR and audio
    if ir_nch > au_nch:
        audio = np.tile(audio[:, np.newaxis], (1, ir_nch))
    elif ir_nch < au_nch:
        ir = np.tile(ir[:, np.newaxis], (1, au_nch))

    # Adjust length so both audio and IR match
    length = au_l + ir_l
    audio = np.pad(audio, pad_width=((0, length - au_l), (0, 0)), mode='constant', constant_values=0)
    ir = np.pad(ir, pad_width=((0, length - ir_l), (0, 0)), mode='constant', constant_values=0)

    result = None
    for c in range(ir_nch):
        if ir_nch > 1:
            ir_chn = ir[:, c]
            au_chn = audio[:, c]
        else:
            ir_chn = ir
            au_chn = audio

        conv = np_fftconvolve(au_chn, ir_chn, mode='full')[:length]

        if result is None:
            result = conv
        else:
            result = np.column_stack((result, conv))

    result = lerp(audio, result, wet)

    return result


def trim_end(data: np.ndarray, trim_db: float = -120, fade_db: float = -96, min_silence: int | None = 512,
             min_length: int | None = None) -> np.ndarray:
    """
    Remove trailing silence
    :param data: Input audio
    :param trim_db: Silence threshold in dB
    :param fade_db: Fade threshold in dB, value should be higher than trim_db
    :param min_silence: Minimum silence length in samples for 1st pass
    :param min_length: Minimum length in samples
    :return: processed audio
    """
    if trim_db is None:
        return data

    nch = data.ndim
    if nch > 1:
        mono = data.mean(axis=1)
    else:
        mono = data

    # Trim 1st pass - trim using peak envelope
    if min_silence is not None:
        hop = 512
        window = hop * 2

        steps = np.arange(math.ceil(len(mono) / hop))
        y = np.array([peak(mono[i * hop:i * hop + window]) for i in steps])  # windowed peak

        th = y > db_to_lin(trim_db)  # Thresholding
        mn = math.ceil(min_silence / hop)
        th_mx = np.array([np.max(th[i:i + mn]) for i in steps])  # Blob small silences by applying windowed max
        # Detect cue indices by finding where the values change and rescale result to original length
        cues = np.clip(np.where(np.diff(th_mx, prepend=False, append=False))[0] * hop, 0, len(mono) - 1)
        regions = cues.reshape(-1, 2)  # Pair cues to get regions

        if len(regions):
            # Use end of 1st region or end of 2nd last region if more than 2 are found
            end = regions[-min(len(regions), 2)][1]
            mono = mono[:end + 1]

    length = len(mono)

    # Trim 2nd pass - tighten trimming
    idx = np.where(np.abs(mono) > db_to_lin(trim_db))[0]
    if len(idx):
        length = idx[-1] + 1

    if min_length is not None:
        length = max(min(len(mono), min_length), length)

    data = data[:length]

    # Apply pseudo-log fade out
    if fade_db is not None:
        idx = np.where(np.abs(mono) > db_to_lin(fade_db))[0]
        if len(idx):
            fade_cue = idx[-1]
            fo = np.append(np.ones(fade_cue), np.linspace(1, 0, length - fade_cue))
            fo = np_log(fo)
            if nch > 1:
                fo = fo.reshape(-1, 1)
                fo = np.repeat(fo, nch, axis=1)
            data *= fo

    return data


def half_cosine(x: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * x)


def sweep_window(length: int, fade: int = 512) -> np.ndarray:
    f_l = min(fade, length // 2)
    x = half_cosine(np.linspace(0, 1, f_l))
    result = np.concatenate([x, np.ones(length - 2 * f_l), x[::-1]])
    return result


def freq_to_period(f: float | np.ndarray, sr: int = 48000) -> int | np.ndarray:
    return np.round(sr / np.array(f)).astype(np.int16)


def lerp(a, b, x):
    return a + (b - a) * x


def db_to_lin(db):
    return np.power(10, db / 20)


def lin_to_db(lin):
    return 20 * np.log10(lin)


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def peak(x):
    return np.max(np.abs(x))


def np_log(array: np.ndarray) -> np.ndarray:
    """
    Pseudo log function for fades
    :param array:
    :return:
    """
    return np.subtract(1, np.subtract(1, array) ** 4)


# Numpy implementations of scipy.signal functions

def np_fftconvolve(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    Perform fft convolution of array a by array b
    :param a:
    :param b:
    :param mode: 'full', 'same' or 'valid'
    :return: Convolved result
    """
    n = a.size + b.size - 1
    a_fft = fft(a, n=n)
    b_fft = fft(b, n=n)
    result_fft = a_fft * b_fft
    result = ifft(result_fft).real

    match mode:
        case 'full':
            return result
        case 'same':
            start = (len(result) - len(a)) // 2
            return result[start:start + len(a)]
        case 'valid':
            valid_size = len(a) - len(b) + 1
            return result[len(b) - 1:len(b) - 1 + valid_size]
        case _:
            raise ValueError("mode must be 'full', 'same', or 'valid'")


def np_decimate(x: np.ndarray, q: int, n: int | None = None) -> np.ndarray:
    """
    Decimate x by an integer factor q using a simple low-pass filter.
    :param x: input signal
    :param  q: Decimate factor
    :param n: moving average filter size, default is 8 times decimate factor
    :return: Decimated result
    """
    if n is None:
        n = 8 * q
    kernel = np.ones(n) / n
    result = np.convolve(x, kernel, mode='same')
    return result[::q]
