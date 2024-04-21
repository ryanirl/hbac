import numpy as np
import torchaudio
import math

from scipy.signal import butter
from scipy.signal import filtfilt

spectrogram = torchaudio.transforms.Spectrogram(
    n_fft = 800, 
    win_length = 256, 
    hop_length = 44, 
    power = None
)


def mad(signal: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis = axis, keepdims = keepdims)
    mad = np.median(np.abs(signal - median), axis = axis, keepdims = keepdims)
    return mad * 1.4826  # This is a constant for normal distribution


def butter_filter(
    signal: np.ndarray, 
    fs: int = 200, 
    cutoff_freq: np.ndarray = np.array([0.25, 50]), 
    order: int = 4, 
    btype: str = "bandpass"
) -> np.ndarray:
    """ 
    """
    b, a = butter(
        N = order, 
        Wn = cutoff_freq / (0.5 * fs), 
        btype = btype,
        analog = False
    )
    return filtfilt(b, a, signal)


def bin_array(
    array: np.ndarray, 
    bin_size: int, 
    axis: int = -1, 
    pad_dir: str = "symmetric", 
    mode: str = "edge", 
    **kwargs
) -> np.ndarray:
    """Given an array and bin size, bins the array along an arbitrary axis into
    bins of size `bin_size`. It will perform padding if the array does not split
    up into equal bin sizes. 

    Args:
        array (np.ndarray): The input array.
        bin_size (int): The size of each bin.
        axis (int): The axis to bin the array along. Default is -1.
        pad_dir (str): The padding direction. One of `left`, `right`, or
            `symmetric` (default).
        mode (str): The padding mode. See the NumPy documentation for options. 

    Returns:
        np.ndarray: The binned array where the number of bins is first. That is,
            the shape will be `(..., n_bins, bin_size, ...)`.

    """
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    # Perform padding if curr_len does not equal new_len.
    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode = mode, **kwargs) # type: ignore

    array = array.reshape(new_shape)
    return array


