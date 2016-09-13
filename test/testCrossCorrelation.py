# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:00:36 2016

@author: dkriete
"""
import numpy as np
from scipy.signal import correlate
from scipy.signal.spectral import _spectral_helper
import matplotlib.pyplot as plt

# Take correlation between two 10 kHz sine waves with 5 us offset
fs = 2e6 # sampling frequency
T = 400e-6 # 400 us sampling period
t = np.arange(int(fs * T)) / fs # array of time points

f = 10e3 # sine wave frequency
dt = 5e-6 # time offset
x = np.sin(2 * np.pi * f * t)
y = np.sin(2 * np.pi * f * (t + dt))

# Calculate cross correlation using integral definition
cc_integral = correlate(x, y) / len(x)
cc_timeaxis = np.linspace(-(len(x) - 1), len(x) - 1, 2 * len(x) - 1) / fs

# Calculate cross correlation using fft method
freqs, times, csd = _spectral_helper(x, y, fs=fs, window='hann', 
                                     nperseg=len(x), detrend=None, 
                                     scaling='density', mode='psd')

# Plot results
fig1 = plt.figure()
ax1 = fig1.add_subplot(111) # for some reason this is preferred over fig.gca()
ax1.set_xlim([-100e-6, 100e-6])
ax1.set_xlabel('Time delay')
ax1.set_ylabel('Cross correlation')
ax1.set_title('Integral method')
ax1.plot(cc_timeaxis, cc_integral)

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.set_xlim([-100e-6, 100e-6])
#ax2.set_xlabel('Time delay')
#ax2.set_ylabel('Cross correlation')
#ax2.set_title('FFT method')
#ax2.plot(cc_timeaxis, cc_fft)