from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pya

from ..exceptions import *

class PeakMatcher(ABC):
    """Abstract class for Peak Matcher implementations.

    A peak matcher handles incoming peaks and assembles them into a suitable
    representation. See `SimplePeakMatcher` and `TrackingPeakMatcher` for
    exemplary implementations.

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def __init__(self, n_frames:int, **kwargs) -> None:
        """Initialize a peak matcher

        Args:
            n_frames (int): The number of frames that the analysis will have.
        """
        pass

    @abstractmethod
    def insert(t: int, t_prev: int, 
               peak_freqs: np.ndarray, peak_coefs: np.ndarray, 
               pitch: float) -> None:
        """Handle a new set of peaks. Peaks can be inserted into the internal
        representation, or discarded. If the insertion of peaks depends on the peaks
        inserted in the previous frame, the parameter t_prev should be respected.
        This allows for more complex stepping procedures that backtrack, run backwards, etc.

        Args:
            t (int): The frame index at which to insert the peaks
            t_prev (int): The frame index to consider as the previous frame. Will be t+1 or t-1
            peak_freqs (np.ndarray): The frequencies of the peaks to be inserted
            peak_coefs (np.ndarray): The coefficients of the peaks to be inserted
            pitch (float): A pitch estimate that can be used to inform the insertion.
                           Can be ignored if it is not relevant to the procedure that
                           is being implemented.
        """
        pass

    @abstractmethod
    def get():
        """Returns the finished representation of the peaks.

        Returns:
            Any type that is appropriate for the type of analysis being done.
        """


class SimplePeakMatcher(PeakMatcher):
    """Implements the simple peak matcher outlined in the report.
    Assembles a constant-dimensional frequency and coefficient matrix that
    track the `n_peaks` first overtones.
    """
    def __init__(self, n_frames, n_peaks=40):
        self.n_peaks = n_peaks
        self.freqs = np.full((n_frames, n_peaks), dtype=float, fill_value=np.nan)
        self.coefs = np.full((n_frames, n_peaks), dtype=complex, fill_value=np.nan)

    def insert(self, t, t_prev, peak_freqs: np.ndarray, peak_coefs: np.ndarray, pitch):
        peak_freqs = peak_freqs.copy()
        peak_coefs = peak_coefs.copy()
        expected_freqs = np.arange(self.n_peaks) * pitch

        if len(peak_freqs) == 0:
            return

        for i, expected_freq in enumerate(expected_freqs):
            stdev = 5 + 0.005*expected_freq

            P = np.exp(-(expected_freq - peak_freqs)**2 / stdev**2)
            P[np.abs(expected_freq - peak_freqs) > 2*stdev] = 0.0

            match_ix = np.argmax(P)

            # if no peak was found for the expected freq
            if P[match_ix] == 0.0:
                continue

            self.freqs[t, i] = peak_freqs[match_ix]
            self.coefs[t, i] = peak_coefs[match_ix]

            # remove peak so that it is not used twice
            peak_freqs[match_ix] = -np.inf 

    def get(self):
        return self.freqs, self.coefs


class TrackingPeakMatcher(PeakMatcher):
    """Implements the tracking peak matcher outlined in the report.
    Assembles a frequency and a coefficient matrix. Each trajectory gets
    its own row in the frequency and coefficient matrix. 
    (I.e. might return n_frames x 900 matrices if there are 900
    trajectories that start and stop over the course of the analysis.)
    """
    def __init__(self, n_frames,
                 max_freq_dev=10.0, freq_dev_increase=0.01, max_n_peaks=100,
                 min_traj_length=5):

        self.min_traj_length = min_traj_length
        self.max_freq_dev, self.freq_dev_increase = max_freq_dev, freq_dev_increase
        self.max_n_peaks = max_n_peaks

        self.n_frames = n_frames
        self.freqs = []
        self.coefs = []
        self.previously_active_trajs = []

    def insert(self, t, t_prev, peak_freqs: np.ndarray, peak_coefs: np.ndarray, pitch: float):
        assert peak_freqs.ndim == peak_coefs.ndim == 1
        assert abs(t - t_prev) == 1

        # match new peaks to prev peaks.
        traj_magnitudes = {traj_ix: abs(self.coefs[traj_ix][t_prev])
                           for traj_ix in self.previously_active_trajs}
        prev_trajs_sorted = sorted(self.previously_active_trajs, 
                                   key=lambda i: -traj_magnitudes[i])
        
        # new_freqs are assumed to be sorted by frequency
        peaks = peak_freqs

        peak_ix             = 0
        n_used_trajectories = 0
        active_trajs        = []
        consumed            = np.zeros(len(peaks), dtype=bool)

        if len(peaks) == 0:
            return

        for traj_ix in prev_trajs_sorted:
            traj_freq = self.freqs[traj_ix][t_prev]
            peak_ix = np.searchsorted(peaks, traj_freq)

            # if we are at the bounds of the new freq array
            if peak_ix == 0:
                look_left  = False
                look_right = not consumed[peak_ix]
            elif 0 < peak_ix < len(peaks):
                look_left  = not consumed[peak_ix-1]
                look_right = not consumed[peak_ix]
            else:
                look_left  = not consumed[peak_ix-1]
                look_right = False

            left_dev  = abs(peak_freqs[peak_ix-1] - traj_freq) if look_left  else np.inf
            right_dev = abs(peak_freqs[peak_ix  ] - traj_freq) if look_right else np.inf
            closest_peak_ix = peak_ix-1 if left_dev < right_dev else peak_ix
            freq_dev = min(left_dev, right_dev)
            dev_threshold = self.max_freq_dev + traj_freq * self.freq_dev_increase

            if freq_dev < dev_threshold:
                self.freqs[traj_ix][t] = peak_freqs[closest_peak_ix]
                self.coefs[traj_ix][t] = peak_coefs[closest_peak_ix]

                # consume peak
                consumed[closest_peak_ix] = True
                active_trajs.append(traj_ix)

                n_used_trajectories += 1
                peak_ix = closest_peak_ix + 1

        # add remaining peaks
        loud_peak_ixs = np.argsort(-np.abs(peak_coefs))
        for i in loud_peak_ixs:
            if consumed[i]:
                continue
            new_traj_ix = len(self.freqs)
            self.freqs.append(np.full(self.n_frames, dtype=float, fill_value=np.nan))
            self.coefs.append(np.full(self.n_frames, dtype=complex, fill_value=np.nan))
            self.freqs[new_traj_ix][t] = peak_freqs[i]
            self.coefs[new_traj_ix][t] = peak_coefs[i]
            active_trajs.append(new_traj_ix)

        self.previously_active_trajs = active_trajs

    def get(self):
        traj_starts = np.zeros(len(self.freqs), dtype=int)
        traj_ends   = np.zeros(len(self.freqs), dtype=int)
        for i, freq in enumerate(self.freqs):
            non_nan_ixs = (~np.isnan(freq)).nonzero()[0]
            traj_starts[i] = non_nan_ixs[0]
            traj_ends[i] = non_nan_ixs[-1] + 1 

        traj_lens        = traj_ends - traj_starts
        trajs_sorted     = traj_lens.argsort()[::-1]
        traj_lens_sorted = traj_lens[trajs_sorted]

        # trajs sorted descending by length, select all that 
        # are longer than our threshold
        i = 0
        for i, length in enumerate(traj_lens_sorted):
            if length < self.min_traj_length:
                break
        # if the last traj was still long enough, include it as well
        if i == len(traj_lens_sorted)-1 and traj_lens_sorted[i] >= self.min_traj_length:
            i += 1

        cutoff = i

        freqs, coefs = np.array(self.freqs), np.array(self.coefs)
        return (
            freqs[trajs_sorted[:cutoff]].transpose(),
            coefs[trajs_sorted[:cutoff]].transpose()
        )


def pack_partials(freqs: np.ndarray, coefs: np.ndarray, 
                  pitches: np.ndarray, n: int):
    """Assemble variable-dimensional peak matrices into a constant-dimensional
    representation based on a fundamental frequency. Can be used to convert
    from a representation built by the TrackingPeakMatcher to a representation
    as the SimplePeakMatcher would return

    Args:
        freqs (np.ndarray): the frequency matrix
        coefs (np.ndarray): the coefficient matrix
        pitches (np.ndarray): the fundamental frequencies (1d array of n_frames)
        n (int): Track the first n overtones.

    Returns:
        a frequencies and a coefs matrix of sizes n_frames x n
    """
    assert (0 <= freqs[~np.isnan(freqs)]).all()
    n_frames, n_trajs = freqs.shape
    ts     = np.arange(n_frames)

    # mask values that exceed the maximum partial
    freqs = freqs.copy()
    max_freq = (pitches * ((n-1) + 0.5)).reshape((-1, 1))
    freqs[freqs >= max_freq] = np.nan

    # calculate multiples of the fundamental freq.
    # mask values that deviate > 0.05 from their expected freq.
    ratios = freqs / pitches.reshape((-1, 1))
    ks     = np.rint(ratios)
    freqs[(ratios - ks) > 0.1] = np.nan

    freqs_packed = np.full((n_frames, n), dtype=float,   fill_value=np.nan)
    coefs_packed = np.full((n_frames, n), dtype=complex, fill_value=np.nan)

    for i in range(n_trajs):
        # find the values for which the trajectory exists and
        # were not masked before.
        val_ixs    = ~np.isnan(freqs[:, i])
        traj_ts    = ts[val_ixs]
        traj_ks    = ks[traj_ts, i].astype(int)
        freqs_packed[traj_ts, traj_ks] = freqs[traj_ts, i]
        coefs_packed[traj_ts, traj_ks] = coefs[traj_ts, i]

    return freqs_packed, coefs_packed


# Use this to register new clip strategies so they can be found
# by the commandline entry points and `get_clip_strategy()`
PEAK_MATCHERS = {
    'simple': SimplePeakMatcher,
    'tracking': TrackingPeakMatcher
}

def get_peak_matcher(matcher: Union[str, PeakMatcher], n_frames=None) -> PeakMatcher:
    """Gets a peak matcher, or returns `matcher` if it is already 
    a PeakMatcher. (For use in parameter sanitation)

    Args:
        strategy (Union[str, PeakMatcher]): 

    Raises:
        ph.StrategyException: If the peak matching strategy is not known.

    Returns:
        PeakMatcher: A matcher of the specified type, or `matcher` if it was
                      already a ClipStrategy.
    """
    if isinstance(matcher, PeakMatcher):
        return matcher
    elif matcher in PEAK_MATCHERS:
        assert n_frames is not None
        return PEAK_MATCHERS[matcher](n_frames)
    else:
        raise StrategyException(f'Unknown peak matcher strategy {matcher}')

