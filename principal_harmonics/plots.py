from typing import Union
from numbers import Real
from functools import partial
from ipywidgets import Output
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pya
import librosa

import principal_harmonics as ph

# used to decorate callbacks whose 
# error output would be swallowed otherwise.
# see https://ipywidgets.readthedocs.io/en/latest/examples/Output%20Widget.html#Debugging-errors-in-callbacks-with-the-output-widget
debug_view = Output()


def _sanitize_note(note: Union[str, int, float]) -> float:
    if isinstance(note, str):
        return librosa.note_to_hz(note)
    elif isinstance(note, int):
        return pya.midicps(note)
    elif isinstance(note, float):
        return note
    else:
        raise TypeError("Note can only be str, int or float")


#------------


def plot_timbre_vectors(freqs: Union[float, np.ndarray], 
                        coefs: np.ndarray, 
                        cmap='plasma', cutoff=-100,
                        colorbar=True):

    if isinstance(freqs, Real):
        n_peaks = coefs.shape[1]
        freqs = (freqs * np.arange(n_peaks)).reshape((1, -1))
        freqs = np.broadcast_to(freqs, coefs.shape)

    if coefs.dtype == complex:
        ampls = 2*np.abs(coefs)
    else:
        ampls = coefs[:]

    ax = plt.gca()
    ax.set_facecolor('black')
    dbs = pya.ampdb(ampls)
    dbs[np.isnan(dbs)] = cutoff
    dbs[dbs < cutoff]  = cutoff

    T, n_trajs = freqs.shape
    ts = np.arange(T)
    for i in range(n_trajs):
        plt.scatter(ts, freqs[:, i], c=dbs[:, i], cmap=cmap, s=2)

    if colorbar:
        cbar_ax = plt.colorbar()
        cbar_ax.set_label("Amplitude [dB]")
        plt.clim(cutoff, np.max(dbs))

    plt.xlabel("Frame index")
    plt.ylabel("Frequency in Hz")

    callback = partial(_timbre_view_on_click, freqs, ampls)
    plt.connect('button_press_event', callback)


@debug_view.capture(clear_output=True)
def _timbre_view_on_click(freqs, ampls, evt):
    t = round(evt.xdata)
    T = len(ampls)
    if not (0 <= t <= T-1):
        print("Cannot play timbre: time out of range")
        return
    syn = ph.pvoc.additive_resynth(freqs[t], ampls[t], stride=256, dur=50)
    syn.stereo().play()


#--------------------


def plot_2d_trajectory(X, interval=50):
    assert X.ndim == 2 and X.shape[1] == 2
    x, y = X[:, 0], X[:, 1]
    n = X.shape[0]
    plt.plot(x, y, alpha=0.5)
    for i in range(0, n, interval):
        plt.text(x[i], y[i], i)


def plot_pca_trajectory(X, features=(0, 1), interval=50, 
                        note: Union[str, float]=None, pipeline=None,
                        **kwargs):
    assert X.ndim == 2

    x, y = X[:, features[0]], X[:, features[1]]
    plt.xlabel(f"components[{features[0]}]")
    plt.ylabel(f"components[{features[1]}]")
    n = X.shape[0]
    plt.plot(x, y, alpha=0.5, marker='.', **kwargs)
    for i in range(0, n, interval):
        plt.text(x[i], y[i], i)

    if pipeline:
        plt.ion()
        veclength = X.shape[1]
        callback = partial(_pca_trajectory_on_click, pipeline, features, veclength, note)
        plt.connect('button_press_event', callback)

    

@debug_view.capture(clear_output=True)
def _pca_trajectory_on_click(pipeline, features, veclength, note, evt):
    x, y = evt.xdata, evt.ydata
    latent_vector = np.zeros(veclength)
    latent_vector[features[0]] = x; latent_vector[features[1]] = y
    
    ampls = pipeline.inverse_transform([latent_vector])
    print(ampls)
    fund = _sanitize_note(note)

    resyn = ph.pvoc.additive_resynth(fund, ampls.squeeze(), stride=256, dur=50)
    resyn.norm().fade_in(0.005).fade_out(0.005).stereo().play()


#----------------------


def plot_ampls(ampls, note = None, dbify=False, db_cutoff=-240, write_numbers=True, text_placement: int = 0, includes_dc=True):
    assert ampls.ndim == 2

    if ampls.dtype == complex:
        ampls = np.abs(ampls)
    else:
        ampls = ampls.copy()

    ampls[np.isnan(ampls)] = 0.0

    if dbify:
        dbs = pya.ampdb(ampls)
        dbs[ampls < db_cutoff] = -240.0
        plt.plot(dbs)
        plt.ylabel("dB level")
    else:
        plt.plot(ampls)
        plt.ylabel("Amplitude")

    if write_numbers:
        for i in range(ampls.shape[1]):
            x = (text_placement + i) % ampls.shape[0]
            if dbify:
                plt.text(x=x, y=dbs[x, i], s=str(i))
            else:
                plt.text(x=x, y=ampls[x, i], s=str(i))

    if note:
        if not includes_dc:
            dc = np.zeros(ampls.shape[0]).reshape((-1, 1))
            ampls = np.hstack((dc, ampls))
        callback = partial(_ampl_view_on_click, ampls, note)
        plt.connect('button_press_event', callback)


@debug_view.capture(clear_output=True)
def _ampl_view_on_click(ampls, note: Union[str, int, float], event):
    t = round(event.xdata)
    if not 0 <= t < ampls.shape[0]:
        print("Time out of range")
        return
    note = _sanitize_note(note)
    print(ampls[t])
    resyn = ph.pvoc.additive_resynth(note, ampls[t], stride=256, dur=50)
    resyn.norm().fade_in(0.005).fade_out(0.005).stereo().play()

