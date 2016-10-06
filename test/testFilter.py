# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:00:31 2016

@author: dkriete
"""

from __future__ import division
import copy
from fdp.classes.fdp_globals import FdpError
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, freqz

def filter_signal(signal, fNyq, fmin=None, fmax=None, numfilttaps=None):
    'Band pass filter the input data'
    
    # Check to see if either filter frequency has been set. If not, then
    # don't filter the data
    if fmin is not None or fmax is not None:
        
        # Set default values for unspecified frequencies and convert units
        # of specified frequencies from kHz to Hz
        if fmin is not None and fmax is not None:
            filttype = 'bandpass'
        
        if fmin is None:
            filttype = 'lowpass'
            fmin = fNyq / 2 # Placeholder valid frequency
        else:
            fmin *= 1000
            
        if fmax is None:
            filttype = 'highpass'
            fmax = fNyq / 2 # Placeholder valid frequency
        else:
            fmax *= 1000
        
        # Verify that frequencies are valid
        if fmin <= 0 or fmin >= fNyq:
            raise FdpError('fmin is outside valid range')
        if fmax <= 0 or fmax >= fNyq:
            raise FdpError('fmax is outside valid range')
        if fmax < fmin:
            raise FdpError('fmin is larger than fmax')
        
        # Filter data using FIR filter generated using window
        # method (Hamming window)
        numpnts = len(signal)
        if numfilttaps is None:
            numtaps = 2 * (numpnts // 2) - 1
        else:
            numtaps = numfilttaps
        
        if filttype == 'lowpass':
            h = firwin(numtaps, fmax, nyq=fNyq)
        elif filttype == 'highpass':
            h = firwin(numtaps, fmin, pass_zero=False, nyq=fNyq)
        else: # bandpass
            h = firwin(numtaps, [fmin, fmax], pass_zero=False, nyq=fNyq)
        
        w, b = freqz(h, worN=2000)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w * fNyq / np.pi, 20 * np.log10(abs(b)))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (dB)')
        
        return filtfilt(h, 1.0, signal, padlen=numtaps)
    else:
        return signal

# =============================================================================
# Start of test
# =============================================================================
T = 0.01 # width of time window
fs = 2e6 # sampling frequency
fNyq = fs / 2

t = np.arange(int(fs * T)) / fs # time array
x = np.sin(2*np.pi*100*t)+0.2*np.sin(2*np.pi*1e3*t)+0.1*np.sin(2*np.pi*10e3*t)

fmin = 0.5
fmax = 2.

x_filt = filter_signal(x, fNyq, fmin=fmin, fmax=fmax)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(t,x)
ax1.set_title('Unfiltered')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(t,x_filt)
ax2.set_title('Filtered')