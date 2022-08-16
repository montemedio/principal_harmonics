import pya
import numpy as np
import sklearn.metrics

from dataclasses import dataclass
from typing import Union

from .pitch_estimation import *
from .sinusoidal_analysis import *
from .synth import *
from .clipping import *

def analyze(asig: pya.Asig,
            midi_note: int,
            pitch_mode: str,
            stride: int,
            clip_strategy: Union[str, 'ClipStrategy'],
            **analysis_args) -> tuple['TimbreAnalysisResult', dict]:
    """Convenience wrapper around the analysis for a single sample.

    Parameters:
        asig (pya.Asig): The signal to analyse
        midi_note (int): The expected pitch as a midi value. 
                         Can be None if pitch_mode == 'variable'
        pitch_mode (str): 'variable' for variable pitch, 
                          'constant' to constantify the pitch using 
                          a weighted median operation
        clip_strategy (str | ClipStrategy): A clipping strategy to apply
                                            to the analysis results.

    Raises:
        ValueError: When an unknown pitch mode was specified.

    Returns:
        res: A TimbreAnalysisResult, metrics: A dict of analysis metrics
    """
    pitches = get_pitch(asig, stride)

    metrics = {}

    if pitch_mode == 'constant':
        expected_freq = pya.midicps(midi_note)
        pitches = constant_pitch(pitches, expected_freq)
        metrics['pitch_dev'] = np.abs(pitches - expected_freq) / expected_freq
    elif pitch_mode == 'variable':
        # no further processing needed
        pass
    else:
        raise ValueError("Unknown pitch mode")

    freqs, coefs = sinusoidal_analysis(asig, 
                                       pitches, 
                                       stride=stride, 
                                       **analysis_args)

    clip_strategy = get_clip_strategy(clip_strategy)
    clip_start, clip_end = clip_strategy.clip(coefs)
    assert clip_start < clip_end, "Clip strategy yielded empty signal."

    freqs = freqs[clip_start:clip_end]; coefs = coefs[clip_start:clip_end]
    asig  = asig[clip_start*stride:clip_end*stride]
    metrics['clip_start'] = clip_start * stride
    metrics['clip_end']   = clip_end   * stride

    harm_tmp_     = phase_correct_resynth(freqs, coefs, asig.sr, stride)
    harmonic_asig = _ensure_length(harm_tmp_, asig.samples)
    noise_asig    = asig - harmonic_asig

    metrics['harmonic_r2'] = sklearn.metrics.r2_score(asig.sig, harmonic_asig.sig)

    res = TimbreAnalysisResult(asig, freqs, coefs, 
                               harmonic_asig, noise_asig)
    return res, metrics


def _ensure_length(asig: pya.Asig, length):
    if asig.samples > length:
        return asig[:length]
    else:
        return asig.pad(length - asig.samples, tail=True) 


def estimate_noise_floor(asig: pya.Asig) -> float:
    """Estimates the noise floor of a sample by computing its spectrum
    and calculating the median magnitude.

    Args:
        asig (pya.Asig): The signal for which to compute the noise floor

    Returns:
        noise_floor:  the noise floor in dbs
    """
    stft = asig.to_stft(nperseg=4096, noverlap=2048)
    dbs = pya.ampdb(np.abs(stft.stft) + 1e-12)
    med_dbs = np.median(dbs, axis=0)
    noise_floor = np.median(med_dbs)
    return noise_floor


@dataclass
class TimbreAnalysisResult:
    """Captures all results of a timbre analysis on a single sample.
    * `asig` contains the clipped and normalized source sample
    * `freqs` contains the peak frequencies
    * `coefs` contains the peak coefficients
    * `harmonic_asig` contains the Phase-correct resynthesis from the
       frequencies and coefficients $\hat s[t]$
    * `noise_asig` contains the stochastic residue. $e[t]$
    """
    asig:  pya.Asig
    freqs: np.ndarray
    coefs: np.ndarray
    harmonic_asig: pya.Asig
    noise_asig:    pya.Asig

    def as_tuple(self):
        return (
            self.asig, 
            self.freqs, self.coefs, 
            self.harmonic_asig, self.noise_asig
        )

