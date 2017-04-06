from __future__ import print_function, absolute_import, division

import numpy as np
import scipy as sp
import scipy.signal

MAX_MEM_BLOCK = 2**8 * 2**11


def framing(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames. '''
    if len(y) < frame_length:
        raise ValueError('Buffer is too short (n={:d})'
                         ' for frame_length={:d}'.format(len(y), frame_length))
    if hop_length < 1:
        raise ValueError('Invalid hop_length: {:d}'.format(hop_length))
    if not y.flags['C_CONTIGUOUS']:
        raise ValueError('Input buffer must be contiguous.')
    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)
    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = np.lib.stride_tricks.as_strided(y,
        shape=(frame_length, n_frames),
        strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann'):
    """ Modified version from `librosa`, since the `librosa` version create
    different windows from `kaldi` and `sidekit`, we make this version
    compatible to them.

    Short-time Fourier transform (STFT)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix

        a complex-valued matrix D such that:
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    See Also
    --------
    istft : Inverse STFT

    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Window the time series.
    y_frames = framing(y, frame_length=win_length, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=np.complex64,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                     stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = sp.fftpack.fft.fft(
            fft_window * y_frames[:, bl_s:bl_t],
            n=n_fft,
            axis=0)[:stft_matrix.shape[0]].conj()
    return stft_matrix


def plot_spectrogram(x, vad=None, ax=None, colorbar=False,
                     linewidth=0.5):
    '''
    Parameters
    ----------
    x : np.ndarray
        2D array
    vad : np.ndarray, list
        1D array, a red line will be draw at vad=1.
    ax : matplotlib.Axis
        create by fig.add_subplot, or plt.subplots
    colorbar : bool, 'all'
        whether adding colorbar to plot, if colorbar='all', call this
        methods after you add all subplots will create big colorbar
        for all your plots
    path : str
        if path is specified, save png image to given path

    Notes
    -----
    Make sure nrow and ncol in add_subplot is int or this error will show up
     - ValueError: The truth value of an array with more than one element is
        ambiguous. Use a.any() or a.all()

    Example
    -------
    >>> x = np.random.rand(2000, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(2, 2, 1)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 2)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 3)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 4)
    >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
    >>> plt.show()
    '''
    from matplotlib import pyplot as plt

    # colormap = _cmap(x)
    colormap = 'spectral'

    if x.ndim > 2:
        raise ValueError('No support for > 2D')
    elif x.ndim == 1:
        x = x[:, None]

    if vad is not None:
        vad = np.asarray(vad).ravel()
        if len(vad) != x.shape[1]:
            raise ValueError('Length of VAD must equal to signal length, but '
                             'length[vad]={} != length[signal]={}'.format(
                                 len(vad), x.shape[1]))
        # normalize vad
        vad = np.cast[np.bool](vad)

    ax = ax if ax is not None else plt.gca()
    ax.set_aspect('equal', 'box')
    # ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title(str(x.shape), fontsize=6)
    img = ax.pcolorfast(x, cmap=colormap, alpha=0.9)
    # ====== draw vad vertical line ====== #
    if vad is not None:
        for i, j in enumerate(vad):
            if j: ax.axvline(x=i, ymin=0, ymax=1, color='r', linewidth=linewidth,
                             alpha=0.3)
    # plt.grid(True)

    if colorbar == 'all':
        fig = ax.get_figure()
        axes = fig.get_axes()
        fig.colorbar(img, ax=axes)
    elif colorbar:
        plt.colorbar(img, ax=ax)

    return ax
