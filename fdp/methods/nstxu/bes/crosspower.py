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
             tmin=0.5, tmax=0.55, nperseg=None, degrees=True, fmin=None,
             fmax=None):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    sig1 = getattr(container, sig1name)
    sig2 = getattr(container, sig2name)
    cs = CrossSignal(sig1, sig2, tmin=tmin, tmax=tmax, nperseg=nperseg,
                     offsetminimum=True, normalizetodc=True, degrees=degrees,
                     fmin=fmin, fmax=fmax)
    return cs


def plotcrosspower(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    spectrum = kwargs.pop('spectrum', False)
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', 200)
    cs = crosssignal(container, *args, **kwargs)
    mask = np.logical_and(fmin <= cs.freqs, cs.freqs <= fmax)
 
    if spectrum:
        logcrosspower = 10*np.log10(cs.crosspower[mask,:])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(cs.times,
                cs.freqs[mask],
                logcrosspower,
                cmap=plt.cm.YlGnBu)
        pcm.set_clim([logcrosspower.max()-100, logcrosspower.max()-20])
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
        logcrosspower = 10*np.log10(cs.crosspower_binavg[mask])
#        logstdevupper = 10*np.log10(cs.crosspower_binavg[mask]
#                      + np.sqrt(cs.crosspower_var[mask]))
#        logstdevlower = 10*np.log10(cs.crosspower_binavg[mask]
#                      - np.sqrt(cs.crosspower_var[mask]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cs.freqs[mask], logcrosspower)
#        ax.fill_between(cs.freqs[mask], logstdevlower, logstdevupper,
#                        alpha=0.5, linewidth=0)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel(r'$10\,\log_{10}(Crosspower)$ $(V^2/Hz)$')
        ax.set_title('{} -- {} -- {}/{} -- Crosspower'.format(
                container.shot,
                container._name.upper(),
                cs.signal1name.upper(),
                cs.signal2name.upper()))
    
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
    mask = np.logical_and(fmin <= cs.freqs, cs.freqs <= fmax)
 
    if spectrum:
        crossphase = cs.crossphase[mask,:]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(cs.times,
                cs.freqs[mask],
                crossphase,
                cmap=plt.cm.RdBu)
        pcm.set_clim([-50, 300])
        cb = plt.colorbar(pcm, ax=ax)
        cb.set_label(r'Angle (' + units + ')')
        ax.set_ylim([fmin,fmax])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (kHz)')
        ax.set_title('{} -- {} -- {}/{} -- Crossphase'.format(
                container.shot,
                container._name.upper(),
                cs.signal1name.upper(),
                cs.signal2name.upper()))
    else:
        crossphase = cs.crossphase_binavg[mask]
#        stdev = np.sqrt(cs.crossphase_var[mask])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cs.freqs[mask], crossphase)
#        ax.fill_between(cs.freqs[mask], crossphase-stdev, crossphase+stdev,
#                        alpha=0.5, linewidth=0)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Angle (' + units + ')')
        ax.set_title('{} -- {} -- {}/{} -- Crossphase'.format(
                container.shot,
                container._name.upper(),
                cs.signal1name.upper(),
                cs.signal2name.upper()))
    return cs

def plotcoherence(container, *args, **kwargs):
    if not isContainer(container):
        warn("Method valid only at container-level", FdpWarning)
        return
    fmin = kwargs.get('fmin', 0)
    fmax = kwargs.get('fmax', 200)
    cs = crosssignal(container, *args, **kwargs)
    mask = np.logical_and(fmin <= cs.freqs, cs.freqs <= fmax)
    coherence = cs.coherence[mask]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cs.freqs[mask], coherence)
    ax.plot((fmin, fmax),(cs.minsigcoh, cs.minsigcoh), 'k-')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_title('{} -- {} -- {}/{} -- Coherence'.format(
            container.shot,
            container._name.upper(),
            cs.signal1name.upper(),
            cs.signal2name.upper()))
    
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
        ax.plot(cs.time_delays * 1000000., cs.correlation_coef_envelope)
        ax.plot((0,0),(0,1), 'k-')
    else:
        ax.plot(cs.time_delays * 1000000., cs.correlation_coef)
        ax.plot((0,0),(-1,1), 'k-')
    ax.set_xlabel('Time delay (us)')
    ax.set_title('{} -- {} -- {}/{} -- Time-Lag Cross Correlation'.format(
            container.shot,
            container._name.upper(),
            cs.signal1name.upper(),
            cs.signal2name.upper()))
    return cs