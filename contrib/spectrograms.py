# Copyright 2022 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio spectrogram functions."""

import dataclasses

# for PyTorch spectrogram
import torch
from torchaudio.transforms import MelSpectrogram
import librosa
import numpy as np

# this is to suppress a warning from torch melspectrogram
import warnings
warnings.filterwarnings("ignore")

# NB: TensorFlow is imported LAZILY inside _compute_logmel (only when
# use_tf_spectral_ops=True). Importing TF at module load alongside PyTorch
# segfaults in this env (TF tries to dlopen CUDA libs that clash with torch's),
# which is exactly why the default path uses the torch/librosa mel front end.

# The TF mel front end was previously imported from `ddsp.spectral_ops`. The
# `ddsp` pip package pulls in `crepe`, whose old setup.py fails to build
# ("No module named 'pkg_resources'"), making this env uninstallable. ddsp's
# compute_logmel is itself pure tf.signal, so we vendor it verbatim from
# ddsp 3.3.4 (spectral_ops.py / core.py). Same ops -> identical features; we
# just drop the unbuildable dependency. (t5/seqio are kept -- they are still
# used by contrib/vocabularies.py.)


def _tf_float32(x):
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)
    return tf.convert_to_tensor(x, tf.float32)


def _stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    audio = _tf_float32(audio)
    if len(audio.shape) == 3:
        audio = tf.squeeze(audio, axis=-1)
    return tf.signal.stft(
        signals=audio,
        frame_length=int(frame_size),
        frame_step=int(frame_size * (1.0 - overlap)),
        fft_length=None,
        pad_end=pad_end)


def _compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    return _tf_float32(tf.abs(_stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end)))


def _compute_mel(audio, lo_hz=0.0, hi_hz=8000.0, bins=64, fft_size=2048,
                 overlap=0.75, pad_end=True, sample_rate=16000):
    mag = _compute_mag(audio, fft_size, overlap, pad_end)
    num_spectrogram_bins = int(mag.shape[-1])
    linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        bins, num_spectrogram_bins, sample_rate, lo_hz, hi_hz)
    mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
    mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
    return mel


def _safe_log(x, eps=1e-5):
    safe_x = tf.where(x <= 0.0, eps, x)
    return tf.math.log(safe_x)


def _compute_logmel(audio, lo_hz=80.0, hi_hz=7600.0, bins=64, fft_size=2048,
                    overlap=0.75, pad_end=True, sample_rate=16000):
    # Lazy TF import (see note at top): only load TF when this TF path is
    # actually requested, and keep it off the GPU (PyTorch owns the GPU).
    global tf
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    mel = _compute_mel(audio, lo_hz, hi_hz, bins, fft_size, overlap, pad_end, sample_rate)
    return _safe_log(mel)


class spectral_ops:  # shim so existing `spectral_ops.compute_logmel(...)` calls work
    compute_logmel = staticmethod(_compute_logmel)

# defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# fixed constants; add these to SpectrogramConfig before changing
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS
    use_tf_spectral_ops: bool = False

    @property
    def abbrev_str(self):
        s = ''
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += 'sr%d' % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += 'hw%d' % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += 'mb%d' % self.num_mel_bins
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def split_audio(samples, spectrogram_config):
    """Split audio into frames."""
    if spectrogram_config.use_tf_spectral_ops:
        # print("split TF")
        return tf.signal.frame(
            samples,
            frame_length=spectrogram_config.hop_width,
            frame_step=spectrogram_config.hop_width,
            pad_end=True)
    else:
        # print("split PT")
        if samples.shape[0] % spectrogram_config.hop_width != 0:
            samples = np.pad(
                samples, 
                (0, spectrogram_config.hop_width - samples.shape[0] % spectrogram_config.hop_width), 
                'constant',
                constant_values=0
            )
        return librosa.util.frame(
            samples,
            frame_length=spectrogram_config.hop_width,
            hop_length=spectrogram_config.hop_width,
            axis=-1).T

def pad_end(samples, n_fft, hop_size):
    """Pad the waveform to ensure that all samples are processed."""
    n_samples = samples.shape[-1]
    # using double negatives to round up
    n_frames = -(-n_samples // hop_size)
    pad_samples = max(0, n_fft + hop_size * (n_frames - 1) - n_samples)
    return torch.nn.functional.pad(samples, (0, pad_samples))

def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    safe_x = torch.where(x <= 0.0, eps, x)
    return torch.log(safe_x)

def compute_spectrogram(
    samples, 
    spectrogram_config,
):
    """
    Compute a mel spectrogram.
    Due to multiprocessing issues running TF and PyTorch together, we use librosa
    and only keep `spectral_ops.compute_logmel` for evaluation purposes.
    """
    if spectrogram_config.use_tf_spectral_ops:
        # NOTE: we only keep this for evaluating existing models
        # This is because I find even with an equivalent PyTorch / librosa implementation 
        # that gives close-enough results (melspec MAE ~ 2e-3), the model output is still affected badly.
        # lazy load
        # print("spec TF")
        overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
        return spectral_ops.compute_logmel(
            samples,
            bins=spectrogram_config.num_mel_bins,
            lo_hz=MEL_LO_HZ,
            overlap=overlap,
            fft_size=FFT_SIZE,
            sample_rate=spectrogram_config.sample_rate)
    else:
        # print("spec PT")
        transform = MelSpectrogram(
            sample_rate=spectrogram_config.sample_rate,
            n_fft=FFT_SIZE,
            hop_length=spectrogram_config.hop_width,
            n_mels=spectrogram_config.num_mel_bins,
            f_min=MEL_LO_HZ,
            f_max=7600,
            power=1.0,
            center=False
        )
        samples = torch.from_numpy(samples).float()
        S = transform(pad_end(samples, FFT_SIZE, spectrogram_config.hop_width))
        S = safe_log(S)
        # S[S<0] = 0
        # S = torch.log(S + 1e-6)
        return S.numpy().T


def flatten_frames(frames, use_tf_spectral_ops=False):
    """Convert frames back into a flat array of samples."""
    if use_tf_spectral_ops:
        # print("flatten TF")
        return tf.reshape(frames, (-1,))
    else:
        # print("flatten PT")
        return np.reshape(frames, (-1,))


def input_depth(spectrogram_config):
    return spectrogram_config.num_mel_bins
