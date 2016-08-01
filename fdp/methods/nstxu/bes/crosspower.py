# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:43:23 2016

@author: drsmith
"""

from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

from fdp.classes.bisignal import Bisignal
from fdp.classes.utilities import isContainer
from fdp.classes.fdp_globals import FdpWarning


def bisignal(container, sig1name='ch01', sig2name='ch02',
             tmin=0.5, tmax=0.55, nperseg=2048):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    sig1 = getattr(container, sig1name)
    sig2 = getattr(container, sig2name)
    bs = Bisignal(sig1, sig2, tmin=tmin, tmax=tmax, nperseg=nperseg,
                  offsetminimum=True, normalizetodc=True)
    return bs


def plotcrosspower(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    bs = bisignal(container, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bs.freqs, 10*np.log10(bs.crosspower))
    ax.set_xlim(0, 200)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Crosspower')
    ax.set_title('{} -- {} -- {}/{} -- Crosspower'.format(
                 container.shot, container._name.upper(),
                 bs.signal1name.upper(), bs.signal2name.upper()))


def plotcoherence(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    bs = bisignal(container, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bs.freqs, bs.cohere)
    ax.set_xlim(0, 200)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Coherence')
    ax.set_title('{} -- {} -- {}/{} -- Coherence'.format(
                 container.shot, container._name.upper(),
                 bs.signal1name.upper(), bs.signal2name.upper()))
