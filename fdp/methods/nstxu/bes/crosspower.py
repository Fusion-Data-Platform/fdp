# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:43:23 2016

@author: drsmith
"""

from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

from ....classes.crosssignal import CrossSignal
from ....classes.utilities import isContainer
from ....classes.fdp_globals import FdpWarning


def crosssignal(container, sig1name='ch01', sig2name='ch02',
                tmin=0.5, tmax=0.55, window='hann', nperseg=None,
                forcepower2=False, offsetminimum=True, offsetdc=False,
                normalizetodc=True, degrees=True, fmin=None, fmax=None,
                numfilttaps=None, removesawteeth=False):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    sig1 = getattr(container, sig1name)
    sig2 = getattr(container, sig2name)
    cs = CrossSignal(sig1, sig2, tmin=tmin, tmax=tmax, window=window,
                     nperseg=nperseg, forcepower2=forcepower2,
                     offsetminimum=offsetminimum, offsetdc=offsetdc,
                     normalizetodc=normalizetodc, degrees=degrees, fmin=fmin,
                     fmax=fmax, numfilttaps=numfilttaps,
                     removesawteeth=removesawteeth)
    return cs


def plotcrosspower(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    spectrum = kwargs.pop('spectrum', False)
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', 200)
    cs = crosssignal(container, *args, **kwargs)
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 200
    mask = np.logical_and(fmin <= cs.freqs, cs.freqs <= fmax)

    if spectrum:
        logcrosspower = 10 * np.log10(cs.crosspower[mask, :])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(cs.times,
                            cs.freqs[mask],
                            logcrosspower,
                            cmap=plt.cm.YlGnBu)
        pcm.set_clim([logcrosspower.max() - 100, logcrosspower.max() - 20])
        cb = plt.colorbar(pcm, ax=ax)
        cb.set_label(r'$10\,\log_{10}(Crosspower)$ $(V^2/Hz)$')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title('{} -- {} -- {}/{} -- Crosspower'.format(
            container.shot,
            container._name.upper(),
            cs.signal1name.upper(),
            cs.signal2name.upper()))
    else:
        crosspower = cs.crosspower_binavg[mask]
        stdev = cs.crosspower_error[mask]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cs.freqs[mask], crosspower, 'k-')
        ax.fill_between(cs.freqs[mask], crosspower - stdev,
                        crosspower + stdev, alpha=0.5, linewidth=0,
                        facecolor='black')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel(r'$10\,\log_{10}(Crosspower)$ $(V^2/Hz)$')
        ax.set_title('{} -- {} -- {}/{} -- Crosspower'.format(
            container.shot,
            container._name.upper(),
            cs.signal1name.upper(),
            cs.signal2name.upper()))
    plt.tight_layout()
    return cs


def plotcrossphase(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    spectrum = kwargs.pop('spectrum', False)
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', 200)
    degrees = kwargs.get('degrees', True)
    if degrees:
        units = 'degrees'
    else:
        units = 'radians'

    cs = crosssignal(container, *args, **kwargs)
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 200
    mask = np.logical_and(fmin <= cs.freqs, cs.freqs <= fmax)

    if spectrum:
        crossphase = cs.crossphase[mask, :]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(cs.times,
                            cs.freqs[mask],
                            crossphase,
                            cmap=plt.cm.RdBu)
        pcm.set_clim([-50, 300])
        cb = plt.colorbar(pcm, ax=ax)
        cb.set_label(r'Angle (' + units + ')')
        ax.set_ylim([fmin, fmax])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title('{} -- {} -- {}/{} -- Crossphase'.format(
            container.shot,
            container._name.upper(),
            cs.signal1name.upper(),
            cs.signal2name.upper()))
    else:
        crossphase = cs.crossphase_binavg[mask]
        stdev = cs.crossphase_error[mask]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cs.freqs[mask], crossphase, 'k-')
        ax.fill_between(cs.freqs[mask], crossphase - stdev, crossphase + stdev,
                        alpha=0.5, linewidth=0, facecolor='black')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Angle (' + units + ')')
        ax.set_title('{} -- {} -- {}/{} -- Crossphase'.format(
            container.shot,
            container._name.upper(),
            cs.signal1name.upper(),
            cs.signal2name.upper()))
    plt.tight_layout()
    return cs


def plotcoherence(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', 200)
    cs = crosssignal(container, *args, **kwargs)
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 200
    mask = np.logical_and(fmin <= cs.freqs, cs.freqs <= fmax)
    coherence = cs.coherence[mask]
    stdev = cs.coherence_error[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cs.freqs[mask], coherence, 'k-')
    ax.plot((fmin, fmax), (cs.minsig_coherence, cs.minsig_coherence), 'k--')
    ax.fill_between(cs.freqs[mask], coherence - stdev, coherence + stdev,
                    alpha=0.5, linewidth=0, facecolor='black')
    ax.set_ylim([0, 1])
    ax.set_xlabel('Frequency (kHz)')
    ax.set_title('{} -- {} -- {}/{} -- Coherence'.format(
        container.shot,
        container._name.upper(),
        cs.signal1name.upper(),
        cs.signal2name.upper()))


<< << << < HEAD

== == == =
    plt.tight_layout()
>>>>>> > master
    return cs


def plotcorrelation(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    envelope = kwargs.pop('envelope', False)
    cs = crosssignal(container, *args, **kwargs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if envelope:
        ax.plot(cs.time_delays * 1000000., cs.correlation_coef_envelope, 'k-')
        ax.plot((0, 0), (0, 1), 'k-')
    else:
        ax.plot(cs.time_delays * 1000000., cs.correlation_coef, 'k-')
        ax.plot((0, 0), (-1, 1), 'k-')
    ax.set_xlim([-250, 250])
    ax.set_xlabel('Time delay (us)')
    ax.set_title('{} -- {} -- {}/{} -- Time-lag cross-correlation'.format(
        container.shot,
        container._name.upper(),
        cs.signal1name.upper(),
        cs.signal2name.upper()))
    plt.tight_layout()
    return cs
