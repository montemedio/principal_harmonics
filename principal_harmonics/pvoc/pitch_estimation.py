import numpy as np
import scipy.stats
import librosa
import pya


def get_pitch(asig: pya.Asig, stride=256, pitch_range: tuple[float, float]=None) -> np.ndarray:
    """Compute a pitch estimate for an audio signal, using the
    YIN pitch tracker.

    Args:
        asig (pya.Asig): The audio signal to compute the pitch for
        stride (int, optional): Hop size of the pitch analysis. Defaults to 256.
        pitch_range (tuple[float, float], optional): Lower and upper limit of the estimated pitch, 
                                                     in Hz. If no limits are passed, the recommended
                                                     defaults (C2 to C7) are used.

    Returns:
        np.ndarray: Pitch estimates for each frame.
    """
    if pitch_range is None:
        # librosa recommended defaults
        fmin, fmax = librosa.note_to_hz(['C2', 'C7'])
    else:
        fmin, fmax = pitch_range
    return librosa.yin(y=asig.sig, fmin=fmin, fmax=fmax, sr=asig.sr,
                       frame_length=4*stride)


def constant_pitch(pitch, expected):
    """Compute a constant fundamental frequency given a sequence of 
    fundamental frequency estimates and and expected fundamental frequency. 
    Uses the weighted median procedure described in the report.
    
    Can be used if `pitch` is unreliable to at least recover a constant
    approximation of the fundamental frequency.

    Args:
        pitch (_type_): The sequence of fundamental frequencies
        expected (_type_): The expected fundamental frequency

    Returns:
        _type_: _description_
    """
    pitch = np.array(pitch); expected = np.array(expected)
    assert pitch.ndim == 1 and expected.ndim == 0

    weights = scipy.stats.norm.pdf(x=pitch,
                                   loc=expected,
                                   scale=0.01*expected)
    detected_fund = weighted_median(pitch, weights)
    return detected_fund


def weighted_median(xs, ws):
    """Compute a weighted median, given data xs and weights ws.
    See https://en.wikipedia.org/wiki/Weighted_median
    """
    return weighted_quantile(xs, ws, 0.5)


def weighted_quantile(xs, ws, quantile):
    """Compute a weighted quantile, given data xs and weights ws.
    See https://en.wikipedia.org/wiki/Weighted_median
    """
    xs = np.array(xs); ws = np.array(ws)
    assert xs.ndim == 1 and ws.ndim == 1
    assert xs.shape == ws.shape
    assert np.all(ws >= 0.0)
    assert len(xs) > 1
    assert 0 < quantile < 1

    sort_ixs = np.argsort(xs)
    xs_sorted = xs[sort_ixs]
    ws_sorted = ws[sort_ixs]
    w_sum = np.cumsum(ws_sorted)
    total_weight = w_sum[-1]
    w_sum /= total_weight

    if total_weight == 0.0:
        raise ValueError("The sum of the weights must not be zero") 
    
    ix = np.searchsorted(w_sum, quantile, side='left')
    if ix == 0:
        return xs_sorted[0]
    elif ix == len(xs)-1:
        return xs_sorted[-1]
    elif w_sum[ix] == quantile:
        return 0.5*(xs_sorted[ix] + xs_sorted[ix+1])
    else:
        return xs_sorted[ix]