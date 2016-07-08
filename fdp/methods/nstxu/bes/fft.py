# -*- coding: utf-8 -*-

from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

from fdp.classes.utilities import isSignal, isContainer
from fdp.classes.fdp_globals import FdpWarning
from fdp.classes.fft import Fft
from . import utilities as UT

def fft(obj, *args, **kwargs):
    """
    Calculate FFT(s) for signal or container.
    Return Fft instance from classes/fft.py
    """
    
    # default to offsetminimum=True for BES ffts
    if 'offsetminimum' in kwargs:
        offsetmin = kwargs.pop('offsetminimum')
    else:
        offsetmin = True
    
    if isSignal(obj):
        return Fft(obj, offsetminimum=offsetmin, *args, **kwargs)
    elif isContainer(obj):
        signalnames = UT.get_signals_in_container(obj)
        ffts = []
        for sname in signalnames:
            signal = getattr(obj, sname)
            ffts.append(Fft(signal, offsetminimum=offsetmin, *args, **kwargs))
        return ffts

def plotfft(signal, fmax=None, *args, **kwargs):
    """
    Plot spectrogram
    """
    if not isSignal(signal):
        warn("Method valid only at signal-level", FdpWarning)
        return
    sigfft = fft(signal, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    pcm = ax.pcolormesh(sigfft.time, 
                        sigfft.freq, 
                        sigfft.psd.transpose(), 
                        cmap=plt.cm.YlGnBu)
    pcm.set_clim([sigfft.psd.max()-100, sigfft.psd.max()-20])
    #ax.set_ylim([0,200])
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(r'$10\,\log_{10}(|FFT|^2)$ $(V^2/Hz)$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_ylim([0,fmax])
    ax.set_title('{} | {} | {}'.format(
                 sigfft.shot, 
                 sigfft.parentname.upper(), 
                 sigfft.signalname.upper()))
    return sigfft

def powerspectrum(signal, fmax=None, *args, **kwargs):
    """
    Calcualte bin-averaged power spectrum
    """
    if not isSignal(signal):
        warn("Method valid only at signal-level", FdpWarning)
        return
    if 'tmin' not in kwargs:
        kwargs['tmin'] = 0.25
    if 'tmax' not in kwargs:
        kwargs['tmax'] = 0.26
    if not fmax:
        fmax = 250
    sigfft = fft(signal, *args, **kwargs)
    psd = np.square(np.absolute(sigfft.fft))
    # bin-averaged PSD, in dB
    sigfft.bapsd = 10*np.log10(np.mean(psd, axis=0))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(sigfft.freq, sigfft.bapsd)
    ax.set_ylabel(r'$10\,\log_{10}(|FFT|^2)$ $(V^2/Hz)$')
    ax.set_xlim([0,fmax])
    ax.set_xlabel('Frequency (kHz)')
    ax.set_title('{} | {} | {} | {}-{} s'.format(
                 sigfft.shot, 
                 sigfft.parentname.upper(), 
                 sigfft.signalname.upper(),
                 kwargs['tmin'],
                 kwargs['tmax']))
    return sigfft
    