# -*- coding: utf-8 -*-

from warnings import warn

import matplotlib.pyplot as plt

from fdp.classes.utilities import isSignal, isContainer
from fdp.classes.fdp_globals import FdpWarning
from fdp.classes.fft import Fft
from . import utilities as UT

def fft(obj, *args, **kwargs):
    """Calculate FFT(s) for signal or container"""
    if isSignal(obj):
        return Fft(obj, *args, **kwargs)
    elif isContainer(obj):
        signalnames = UT.get_signals_in_container(obj)
        ffts = []
        for sname in signalnames:
            signal = getattr(obj, sname)
            ffts.append(Fft(signal, *args, **kwargs))
        return ffts

def plotfft(signal, fmax=None, *args, **kwargs):
    """Calcualte FFT for signal and plot"""
    if not isSignal(signal):
        warn("Method valid only at signal-level", FdpWarning)
        return
    if not fmax:
        fmax = 1e3
    fft = Fft(signal, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    pcm = ax.pcolormesh(fft.time, 
                        fft.freq, 
                        fft.psd.transpose(), 
                        cmap=plt.cm.YlGnBu)
    pcm.set_clim([fft.psd.max()-100, fft.psd.max()-20])
    #ax.set_ylim([0,200])
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r'$10\,\log_{10}(|FFT|^2)$ $(V^2/Hz)$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_ylim([0,fmax])
    ax.set_title('{} | {} | {}'.format(
                 fft.shot, 
                 fft.parentname.upper(), 
                 fft.signalname.upper()))
    return fft