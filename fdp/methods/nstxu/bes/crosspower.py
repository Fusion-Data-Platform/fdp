# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:43:23 2016

@author: drsmith
"""

from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

from fdp.classes.crosssignal import CrossSignal
from fdp.classes.utilities import isContainer
from fdp.classes.fdp_globals import FdpWarning


def crosssignal(container, sig1name='ch01', sig2name='ch02',
             tmin=0.5, tmax=0.55, nperseg=2048):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    sig1 = getattr(container, sig1name)
    sig2 = getattr(container, sig2name)
    bs = CrossSignal(sig1, sig2, tmin=tmin, tmax=tmax, nperseg=nperseg,
                     offsetminimum=True, normalizetodc=True)
    return bs


def plotcrosspower(container, spectrum=False, fmax=200, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    cs = crosssignal(container, *args, **kwargs)

    print spectrum 
    if spectrum:
        print 'spectrum if statement has executed'
        logcrosspower = 10*np.log10(cs.crosspower)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(cs.times,
                cs.freqs,
                logcrosspower,
                cmap=plt.cm.YlGnBu)
        pcm.set_clim([logcrosspower.max()-100, logcrosspower.max()-20])
        cb = plt.colorbar(pcm, ax=ax)
        cb.set_label(r'$10\,\log_{10}(Crosspower)$ $(V^2/Hz)$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_ylim(0, fmax)
        ax.set_title('{} -- {} -- {}/{} -- Crosspower'.format(
                container.shot,
                container._name.upper(),
                cs.signal1name.upper(),
                cs.signal2name.upper()))
    else:
        logcrosspower = 10*np.log10(cs.crossphase_binavg)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cs.freqs, logcrosspower)
        ax.set_xlim(0, fmax)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Crosspower')
        ax.set_title('{} -- {} -- {}/{} -- Crosspower'.format(
                container.shot, container._name.upper(),
                cs.signal1name.upper(), cs.signal2name.upper()))
                 
def plotcrossphase(container, spectrum=False, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    bs = crosssignal(container, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bs.freqs, bs.crossphase)
    ax.set_xlim(0, 200)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Crossphase')
    ax.set_title('{} -- {} -- {}/{} -- Crossphase'.format(
                 container.shot, container._name.upper(),
                 bs.signal1name.upper(), bs.signal2name.upper()))

def plotcoherence(container, spectrum=False, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    bs = crosssignal(container, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bs.freqs, bs.cohere)
    ax.set_xlim(0, 200)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Coherence')
    ax.set_title('{} -- {} -- {}/{} -- Coherence'.format(
                 container.shot, container._name.upper(),
                 bs.signal1name.upper(), bs.signal2name.upper()))
