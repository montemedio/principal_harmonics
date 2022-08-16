from typing import Union
from dataclasses import dataclass
from functools import cache
from numbers import Real

import pya
import scipy.fft
import numpy as np

from .peak_matching import *


def sinusoidal_analysis(asig: pya.Asig,
                        pitch_analysis: Union[float, np.ndarray],
                        n_periods: float = 6, n_peaks: int = 40,
                        fill_value: float = np.nan, stride: int = 256,
                        remove_short_limit=5,
                        interpolate_hole_limit=10,
                        return_steps=False,
                        peak_matching='simple') \
        -> tuple[np.ndarray, np.ndarray]:
    """Main entry point for the sinusoidal analysis.

    Arguments:
        asig (pya.Asig): The audio signal to analyze
        pitch_analysis (np.ndarray | float): The estimated fundamental frequencies for each analysis frame.
                                             Can be either a single float, or a sequence with as many frames
                                             as this analysis will have
        n_periods (float): Multiplier for the window size. Given the current estimated fundamental frequncy f,
                           the window size will be int((n_samples * n_periods) // 2 * 2 + 1) Default: 6
        n_peaks (int): The number of overtones to track. Default: 40
        fill_value (float): The value with which to fill holes. Default: np.nan
        remove_short_limit (int): The maximum amount of frames for which a trajectory can be considered
                                  a false positive and removed from sinusoidal analysis. Default: 5
        interpolate_hole_limit (int): The maximum amount of frames to interpolate over. Default: 10
        return_steps (bool): Whether to return the spectral analysis results. Can be used for debugging
                             purposes.
        peak_matching (str | PeakMatcher): The peak matching strategy to use.

    Returns:
        if not peak_matching:
            returns freqs and coefs matrices
        else:
            returns freqs and coefs matrices and list of `SinusoidalAnalysisStep`s
    """

    n_frames = asig.samples // stride + 1
    fundamentals = _sanitize_timbre_analysis_funds(pitch_analysis, n_frames)

    freqs = np.zeros((n_frames, n_peaks), dtype=float)
    coefs = np.zeros((n_frames, n_peaks), dtype=complex)

    peak_matcher = get_peak_matcher(peak_matching, n_frames)

    analysis_steps = []
    for i, fundamental in enumerate(fundamentals):

        n_samples                    = 1 / fundamental * asig.sr
        window_size                  = int((n_samples * n_periods) // 2 * 2 + 1)
        asig_slice                   = slice_asig(asig, stride, i, window_size)
        current_freqs, current_coefs = compute_spectrum(asig_slice, asig.sr)
        noise_floor                  = estimate_noise_floor(current_coefs) + 6
        peak_freqs, peak_coefs       = detect_peaks(current_freqs, current_coefs, noise_floor + 6)
        
        peak_matcher.insert(t=i, t_prev=i-1, peak_freqs=peak_freqs, 
                            peak_coefs=peak_coefs, pitch=fundamental)

        if return_steps:
            analysis_steps.append(
                SinusoidalAnalysisStep(fundamental, n_samples, window_size, asig_slice,
                                       noise_floor, 
                                       current_freqs, current_coefs,
                                       peak_freqs, peak_coefs)
            )

    freqs, coefs = peak_matcher.get()
    freqs, coefs = cleanup(freqs, coefs, asig.sr, stride, interpolate_hole_limit, remove_short_limit)

    if fill_value != np.nan:
        freqs[np.isnan(freqs)] = fill_value
        coefs[np.isnan(coefs)] = fill_value

    if return_steps:
        return freqs, coefs, analysis_steps
    else:
        return freqs, coefs


@dataclass
class SinusoidalAnalysisStep:
    """Captures all spectral analysis results for a single frame
    of the sinusoidal analysis.
    """
    fundamental: float
    n_samples: int
    window_size: int
    asig_slice: pya.Asig
    noise_floor: float
    fft_freqs: np.ndarray
    fft_coefs: np.ndarray
    peak_freqs: np.ndarray
    peak_coefs: np.ndarray


def slice_asig(asig: pya.Asig,
               stride: int, i: int, window_size: int):
    asig_len = asig.samples

    start = stride*i - window_size // 2
    end   = stride*i + window_size // 2 + 1

    slice_from = max(0, start)
    slice_to = min(asig_len, end)
    asig_slice = asig[slice_from:slice_to]

    # if we reached the end of the signal and it is not long enough, pad
    # the necessary amount
    if start < 0:
        asig_slice = asig_slice.pad(width=abs(start), constant_values=0,
                                    tail=False)
    if end > asig_len:
        asig_slice = asig_slice.pad(width=end - asig_len, constant_values=0,
                                    tail=True)
    return asig_slice

"""
def compute_spectrum(sig: pya.Asig, sr, pad_factor=8):
    M = sig.samples
    N = scipy.fft.next_fast_len(M * pad_factor)
 
    window = _get_cached_window(M)
    window /= np.sum(window)
    windowed_sig = sig*window   
    aspec: pya.Aspec = windowed_sig.pad(N-M).to_spec()

    return aspec.freqs, aspec.rfftspec
"""

def compute_spectrum(sig: pya.Asig, sr, pad_factor=8):
    # Compute spectrum using zero-phase padding, as explained in
    # https://ccrma.stanford.edu/~jos/sasp/Zero_Phase_Zero_Padding.html

    if isinstance(sig, pya.Asig):
        sig: np.ndarray = sig.sig
    if not sig.ndim == 1:
        raise ValueError("Need 1D signal")
    if not len(sig % 2 == 1):
        raise ValueError("Need odd-length signal")

    M = len(sig)
    N = scipy.fft.next_fast_len(M*pad_factor)
    buffer = np.zeros(N)

    window = _get_cached_window(M)
    window /= np.sum(window)
    windowed_sig = sig*window

    negative_time = windowed_sig[:(M-1)//2]
    positive_time = windowed_sig[(M-1)//2:]
    buffer[0:(M+1)//2] = positive_time
    buffer[N-(M-1)//2:N] = negative_time

    freqs = scipy.fft.rfftfreq(len(buffer), 1/sr)
    rfft  = scipy.fft.rfft(buffer)
    return freqs, rfft


@cache
def _get_cached_window(length):
    # Use fftbins=False to get a symmetric window
    return scipy.signal.get_window('blackmanharris', length, fftbins=False)


def detect_peaks(freqs: np.ndarray, coefs: np.ndarray, cutoff_db=-np.inf):
    """Detect peaks in the spectrum, using parabolic interpolation.

    Args:
        freqs (np.ndarray): _description_
        coefs (np.ndarray): _description_
        cutoff_db (_type_, optional): Minimum amplitude for the peaks. Defaults to -np.inf.

    Returns:
        _type_: an ndarray of peak frequencies and a matching ndarray of peak coefficients
    """
    if not freqs.ndim == 1 or not coefs.ndim == 1:
        raise ValueError("Need 1D inputs")
    if not freqs.shape == coefs.shape:
        raise ValueError("freqs and coefs need to have the same shape")

    ampls = 2*np.abs(coefs)
    ampls[ampls < 1e-12] = 1e-12    # divergence protection, clip at -240db
    dbs = pya.ampdb(ampls)
    peak_ixs, _ = scipy.signal.find_peaks(dbs, prominence=0.5, height=cutoff_db)

    include_first = 0            in peak_ixs
    include_last  = len(freqs)-1 in peak_ixs
    peak_ixs = np.delete(peak_ixs, np.where((peak_ixs == 0) | (peak_ixs == len(freqs)-1)))

    if len(peak_ixs) == 0:
        return np.array([]), np.array([], dtype=complex)

    # interpolation adapted from
    # "PARSHL - an analysis/synthesis program for non-harmonic sounds
    # based on sinusoidal representation" by Smith and Serra, p. 13
    a, b, c = dbs[peak_ixs-1], dbs[peak_ixs], dbs[peak_ixs+1]
    peak_offset = 0.5 * (a - c) / (a - 2*b + c)

    # frequency interpolation
    bin_size = freqs[1]
    interp_freqs  = freqs[peak_ixs] + bin_size * peak_offset

    # parabolic interpolation for amplitudes
    interp_mag_db = b - 0.25*(a - c)*peak_offset
    interp_ampls  = pya.dbamp(interp_mag_db)

    # linear interpolation for phases, using the same peak_offset
    a, b, c       = np.angle(coefs[peak_ixs-1]), np.angle(coefs[peak_ixs]), np.angle(coefs[peak_ixs+1])
    phase_jumps   = np.abs(a - b)
    xs            = np.column_stack([peak_ixs-1, peak_ixs, peak_ixs+1]).ravel()
    xs_interp     = peak_ixs + peak_offset
    ys            = np.column_stack([a, b, c]).ravel()

    interp_phases = np.where(phase_jumps < np.pi,
                              # if jump < pi, use interpolated
                             np.interp(xs_interp, xs, ys), 
                              # else, use original
                             b)
    
    interp_coefs  = 0.5*interp_ampls * np.exp(1j * interp_phases)

    res_freqs = interp_freqs
    res_coefs = interp_coefs
    if include_first:
        res_freqs = np.insert(res_freqs, 0, freqs[0])
        res_coefs = np.insert(res_coefs, 0, coefs[0])
    if include_last:
        res_freqs = np.append(res_freqs, freqs[-1])
        res_coefs = np.append(res_coefs, coefs[-1])
    
    return res_freqs, res_coefs


def estimate_noise_floor(coefs: np.ndarray):
    dbs = pya.ampdb(np.abs(coefs) + 1e-12)
    return np.median(dbs)


def cleanup(freqs, coefs, sr, stride, interpolate_limit=5, remove_limit=2):
    """Given a frequency and coefficient matrix, perform removal of short
    trajectories up to `remove_limit` frames and interpolation over holes up to
    `interpolate_limit` frames.
    """
    freqs = freqs.copy()
    ampls = 2*np.abs(coefs)
    phases = np.angle(coefs)

    nans = np.isnan(freqs)

    def remove_short(threshold):
        for short_traj in _find_short_trajs(nans, threshold):
            t1, t2, n = short_traj
            freqs [t1:t2, n] = np.nan
            ampls [t1:t2, n] = np.nan
            phases[t1:t2, n] = np.nan
            nans  [t1:t2, n] = True
    
    def close_holes(limit):
        for hole in _find_interpolation_holes(nans, limit):
            t1, t2, n = hole
            ts_holes = np.arange(t1, t2)
            ts_known = [t1-1, t2]

            freqs[t1:t2, n] = np.interp(ts_holes, ts_known, [freqs[t1-1, n], freqs[t2, n]])
            ampls[t1:t2, n] = np.interp(ts_holes, ts_known, [ampls[t1-1, n], ampls[t2, n]])
            nans[ t1:t2, n] = True

            f1   = freqs[t1-1, n] / sr * 2*np.pi
            f2   = freqs[t2,   n] / sr * 2*np.pi
            phi1 = phases[t1-1, n]
            phi2 = phases[t2,   n]
            phases[t1:t2, n] = _smooth_phase_interp(f1, f2, phi1, phi2, t2-t1+1, stride)

    remove_short(remove_limit)
    close_holes(interpolate_limit)
        
    return freqs, 0.5*ampls * np.exp(phases*1j)

            
#@numba.njit
def _find_interpolation_holes(nans: np.ndarray, limit: int):
    T, N = nans.shape
    t1 = t2 = 0

    for n in range(N):
        t1 = 0

        # seek to the first non-hole value
        while t1 < T:
            if not nans[t1, n]:
                break
            t1 += 1

        while t1 < T:
            # find the beginning of a hole
            while t1 < T:
                if nans[t1, n]:
                    break
                t1 += 1
            else:  # t1 == T
                break
            
            # find the end of the hole
            t2 = t1
            while t2 < T:
                if not nans[t2, n]:
                    break
                t2 += 1
            else:   # t2 == T
                break

            # now: nans[t1:t2, n].all() == True

            if t2 - t1 <= limit:
                yield t1, t2, n

            t1 = t2


def _find_short_trajs(nans: np.ndarray, threshold: int):
    T, N = nans.shape
    t1 = t2 = 0
    for n in range(N):
        t1 = 0

        while t1 < T:
            while t1 < T:
                if not nans[t1, n]:
                    break
                t1 += 1
            else:  # t1 == T
                break

            t2 = t1

            while t2 < T:
                if nans[t2, n]:
                    break
                t2 += 1
            else:  # t2 == T
                break

            if t2 - t1 <= threshold:
                yield t1, t2, n

            t1 = t2


def _smooth_phase_interp(f1, f2, phi1, phi2, S, stride):
    # phase interpolation adapted from McAulay and Quatieri, 1986
    x    = 1/(2*np.pi) * ((phi1 + f1*S - phi2) + S*0.5*(f2 - f1))
    M    = np.rint(x)
    tmp  = phi2 - phi1 - f1*S + 2*np.pi*M
    eta  =  3/(S**2)*tmp - 1/S     *(f2 - f1) 
    iota = -2/(S**3)*tmp + 1/(S**2)*(f2 - f1)

    ms = np.arange(1*stride, S*stride, step=stride)
    return phi1 + f1*ms + eta*ms**2 + iota*ms**3

    
def _sanitize_timbre_analysis_funds(input, n_frames):
    if isinstance(input, Real):
        fundamentals = np.repeat(float(input), n_frames)
    elif isinstance(input, np.ndarray):
        assert input.ndim == 1
        assert input.shape[0] == n_frames
        fundamentals = input
    return fundamentals
