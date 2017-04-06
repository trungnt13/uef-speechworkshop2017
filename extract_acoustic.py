# ===========================================================================
# Required packages:
# * numpy
# * scipy
# * matplotlib
# ===========================================================================
from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import os
import sys

import numpy as np
import scipy as sp
import scipy.signal
import scipy.fftpack as fft
from scipy.io import wavfile

from utils import plot_spectrogram, framing

# ===========================================================================
# Trick to get the right script path, then infer the data path
# ===========================================================================
data_path = os.path.join(os.path.dirname(sys.argv[0]), 'data', 'sample1.wav')
sample_rate, data = wavfile.read(data_path)
# preemphasis
data = np.append(data[0], data[1:] - 0.97 * data[:-1])

print(data, sample_rate)

plt.figure()
plt.plot(data)
# plt.show(block=True)

# ===========================================================================
# Step-by-step SFTF
# ===========================================================================
WIN_LENGTH = int(0.02 * sample_rate)
HOP_LENGTH = int(0.01 * sample_rate)
MAX_MEM_BLOCK = 2**8 * 2**11 # in bytes
print("Maximum memory block:", MAX_MEM_BLOCK / 1024 / 1024, "MB")
N_FFT = 256
WINDOW = 'hann'

fft_window = scipy.signal.get_window(WINDOW, WIN_LENGTH, fftbins=True
        ).reshape(-1, 1)
plt.figure()
plt.plot(fft_window)

y_frames = framing(data, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH)

energy = (y_frames**2).sum(axis=0).astype('float32')
energy = np.where(energy == 0., np.finfo(float).eps, energy)
log_energy = np.log(energy).astype('float32')
plt.figure()
plt.plot(log_energy)

# Pre-allocate the STFT matrix
stft_matrix = np.empty((int(1 + N_FFT // 2), y_frames.shape[1]),
                       dtype=np.complex64,
                       order='F')
# how many columns can we fit within MAX_MEM_BLOCK?
n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                 stft_matrix.itemsize))
for bl_s in range(0, stft_matrix.shape[1], n_columns):
    bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
    # RFFT and Conjugate here to match phase from DPWE code
    stft_matrix[:, bl_s:bl_t] = fft.fft(
        fft_window * y_frames[:, bl_s:bl_t],
        n=N_FFT,
        axis=0)[:stft_matrix.shape[0]].conj()

windowed = fft_window * y_frames[:, bl_s:bl_t]
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(y_frames[:, 0])
plt.title("Original frame")
plt.subplot(1, 2, 2)
plt.plot(windowed[:, 0])
plt.title("Windowed frame")

# Magnitude of the STFT
S = np.abs(stft_matrix)
# power spectrum of the STFT
S = S ** 2
# log-power spectrum
logS = np.log10(S)

# show the log-power spectrum
plt.figure()
plot_spectrogram(logS)
plt.show(block=True)
