import pya
import numpy as np
import scipy.interpolate


def additive_resynth(freqs, ampls, sr=44100, stride=256, dur=None) -> pya.Asig:
    """Perform additive resynthesis (not phase-correct) given frequencies and amplitudes.

    Args:
        freqs: A float, to indicate a constant fundamental frequency.
               Or a 1d array to indicate a set of frequencies for one frame that will be broadcast 
                 over the whole signal.
               Or a 2d array to indicate a different set of frequencies for each frame.
        ampls: Amplitudes matching to `freqs`. The same broadcasting rules apply
               as with `freqs`.
        sr (int, optional): Sampling rate of the resulting signal. Defaults to 44100.
        stride (int, optional): Number of samples  between frames. Defaults to 256.
        dur (_type_, optional): Duration of the signal in frames. Can only be used if
                                `freqs` and `ampls` are numbers, 0d or 1d arrays.
                                Defaults to None.

    Returns:
        pya.Asig: The resynthesized signal.
    """
    freqs, ampls = _sanitize_resynth(freqs, ampls, dur)

    circle_freqs = 2*np.pi * freqs / sr
    n_samples           = circle_freqs.shape[0]
    ts_samples          = np.arange(0, n_samples*stride, step=stride)

    ts                  = np.arange(0, (n_samples-1)*stride, step=1)
    interp_freqs        = scipy.interpolate.interp1d(ts_samples, circle_freqs, 
                                                     assume_sorted=True, axis=0)
    instantaneous_freqs = interp_freqs(ts)
    instantaneous_phases = np.cumsum(instantaneous_freqs, axis=0)

    interp_ampls        = scipy.interpolate.interp1d(ts_samples, ampls, 
                                                     assume_sorted=True, axis=0)
    instantaneous_ampls = interp_ampls(ts)

    sig = np.sum(instantaneous_ampls * np.cos(instantaneous_phases), axis=1)
    resynth_asig = pya.Asig(sig, sr=sr)
    return resynth_asig


def _sanitize_resynth(freqs, ampls, dur=None):
    if isinstance(ampls, np.ndarray) and ampls.dtype == complex:
        raise TypeError("Need real amplitudes, not complex coefficients")

    freqs = np.array(freqs, dtype=float, copy=True)
    ampls = np.array(ampls, dtype=float, copy=True)
    if freqs.ndim > 2 or ampls.ndim > 2:
        raise ValueError("Need 0, 1 or 2D arrays")

    # this is used to enforce the right dimensionality and duration
    if dur is None:
        dur = 1
    tmp = np.zeros(shape=(dur, 1))

    synthesize_overtones = False
    if freqs.ndim in [0, 1]:
        freqs = freqs.reshape((1, -1))
        synthesize_overtones = True
    if ampls.ndim in [0, 1]:
        ampls = ampls.reshape((1, -1))

    print(tmp.shape, freqs.shape, ampls.shape)
    _, broadcast_freqs, broadcast_ampls = np.broadcast_arrays(tmp, freqs, ampls)

    # if we did not get all frequencies but just one
    # fundamental or a fundamental trajectory
    if synthesize_overtones:
        _, n_overtones = broadcast_freqs.shape
        broadcast_freqs = broadcast_freqs * np.arange(n_overtones).reshape((1, -1))

    broadcast_freqs[np.isnan(broadcast_freqs)] = 0.0
    broadcast_ampls[np.isnan(broadcast_ampls)] = 0.0
    return broadcast_freqs, broadcast_ampls


def phase_correct_resynth(freqs, coefs, sr=44100, stride=256, 
                          return_interp_results=False) -> pya.Asig:
    """Perform phase-correct resynthesis using frequencies and complex coefficients.

    Args:
        freqs: A float, 0d, 1d or 2d array. Needs to be broadcastable to `coefs` using numpy's
               broadcasting rules.
        coefs: A 2d array of complex coefficients.
        sr (int, optional): Sampling rate of the resynthesized signal. Defaults to 44100.
        stride (int, optional): Number of samples between frames. Defaults to 256.
        return_interp_results (bool, optional): _description_. Defaults to False.

    Returns:
        pya.Asig: _description_
    """
    freqs, coefs = _sanitize_phase_correct_resynth(freqs, coefs)

    freqs               = freqs / sr  * 2*np.pi # convert to radian frequencies
    n_frames, n_peaks   = freqs.shape
    ts_frames           = np.arange(0, n_frames*stride, step=stride)
    ts                  = np.arange(0, n_frames*stride, step=1)

    nans   = np.isnan(coefs)
    ampls  = 2*np.abs(coefs)

    tmp    = coefs.copy()
    tmp[nans] = 0.0
    phases = np.angle(tmp) % (2*np.pi)
    phases[nans] = np.nan

    # instantaneous_ampls = np.interp(ts, ts_frames, ampls, axis=0)
    # magnitude interpolation
    ampls[np.isnan(ampls)] = 0.0
    instantaneous_ampls = scipy.interpolate.interp1d(ts_frames, ampls, axis=0, 
                                                     bounds_error=False,
                                                     assume_sorted=True, fill_value=0.0)(ts)
    
    prev_phases = phases[ :-1]
    curr_phases = phases[1:  ]
    prev_freqs  = freqs [ :-1]
    curr_freqs  = freqs [1:  ]

    # extrapolate phases: 
    # np.angle(0+0j) == 0.0. This will lead to an incorrect phase interpolation,
    # meaning we would get an incorrect pitch for the oscillator while it
    # is being ramped up from amplitude zero / ramped down to amplitude zero.
    np.putmask(prev_phases,
               mask=( np.isnan(prev_phases) & ~np.isnan(curr_phases)),
               values=curr_phases - curr_freqs*stride)
    np.putmask(curr_phases,
               mask=(~np.isnan(prev_phases) &  np.isnan(curr_phases)),
               values=prev_phases + prev_freqs*stride)

    # extrapolate frequencies:
    # while the oscillator is ramped up or down we do not want a sudden jump
    # of the frequency to zero. Just extrapolate it into the frame where the amplitude
    # is being ramped. The frequency jump will still exist, but amplitude
    # will be zero while it occurs.
    np.putmask(prev_freqs,
               mask=( np.isnan(prev_freqs) & ~np.isnan(curr_freqs)),
               values=curr_freqs)
    np.putmask(curr_freqs,
               mask=(~np.isnan(prev_freqs) &  np.isnan(curr_freqs)),
               values=prev_freqs)

    prev_phases[np.isnan(prev_phases)] = 0.0
    curr_phases[np.isnan(curr_phases)] = 0.0
    prev_freqs[np.isnan(prev_freqs)]   = 0.0
    curr_freqs[np.isnan(curr_freqs)]   = 0.0

    # phase interpolation adapted from McAulay and Quatieri, 1986
    x     = 1/(2*np.pi) * ((prev_phases + prev_freqs*stride - curr_phases) + stride*0.5*(curr_freqs - prev_freqs))
    M     = np.rint(x)
    tmp   = curr_phases - prev_phases - prev_freqs*stride + 2*np.pi*M
    etas  =  3/(stride**2)*tmp - 1/stride     *(curr_freqs - prev_freqs) 
    iotas = -2/(stride**3)*tmp + 1/(stride**2)*(curr_freqs - prev_freqs)

    ms = np.arange(stride).reshape((-1, 1))
    instantaneous_phases = np.zeros((stride*n_frames, n_peaks))
    for i in range(n_frames-1):
        phi, omega, eta, iota = prev_phases[i:i+1], prev_freqs[i:i+1], etas[i:i+1], iotas[i:i+1]
        instantaneous_phases[i*stride:(i+1)*stride] = phi + omega*ms + eta*ms**2 + iota*ms**3

    sig = np.sum(instantaneous_ampls * np.cos(instantaneous_phases), axis=1)
    resynth_asig = pya.Asig(sig, sr=sr)
    
    if return_interp_results:
        return resynth_asig, instantaneous_ampls, instantaneous_phases
    else:
        return resynth_asig


def _sanitize_phase_correct_resynth(freqs, coefs):
    if not coefs.dtype == complex:
        raise TypeError("Need complex-valued coefs for phase correct resynth")

    freqs = np.array(freqs, dtype=float,   copy=True)
    coefs = np.array(coefs, dtype=complex, copy=True)

    if freqs.ndim > 2:
        raise ValueError("Need 0, 1 or 2D freqs")
    if coefs.ndim != 2:
        raise ValueError("Need 2D coefs for _phase-correct_ resynth")

    synthesize_overtones = False
    if freqs.ndim in [0, 1]:
        freqs = freqs.reshape((-1, 1))
        synthesize_overtones = True

    broadcast_freqs, _ = np.broadcast_arrays(freqs, coefs)

    # if we did not get all frequencies but just one
    # fundamental or a fundamental trajectory
    if synthesize_overtones:
        _, n_overtones = broadcast_freqs.shape
        broadcast_freqs = broadcast_freqs * np.arange(n_overtones).reshape((1, -1))

    return broadcast_freqs, coefs

