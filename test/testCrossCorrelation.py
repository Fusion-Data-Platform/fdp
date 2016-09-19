# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:00:36 2016

@author: dkriete
"""
import numpy as np
from scipy.signal import correlate, fftconvolve
from scipy.signal.spectral import _spectral_helper
import matplotlib.pyplot as plt

# Take correlation between two 10 kHz sine waves with 5 us offset
fs = 2e6 # sampling frequency
T = 400e-6 # 400 us sampling period
t = np.arange(int(fs * T)) / fs # array of time points
N = len(t) # Number of points in signals (N=800 for this case)

f = 10e3 # sine wave frequency
dt = 5e-6 # time offset
x = np.sin(2 * np.pi * f * t)
y = np.sin(2 * np.pi * f * (t + dt))

# =============================================================================
# Test stuff written during development of cross-correlation routine
# =============================================================================
# Calculate cross correlation using integral definition
#cc_integral = np.zeros(2 * N - 1)
#for k in range(len(cc_integral)):
#    if k in range(N): # First half is zero and negative time shift
#        for j in range(N - 1 - k, N):
#            cc_integral[k] += x[j] * y[j + k - (N - 1)]
##        cc_integral[k] /= k + 1
#    else: # Second half is positive time shift
#        for j in range(0, 2 * (N - 1) - k + 1):
#            cc_integral[k] += x[j] * y[j + k - (N - 1)]
##        cc_integral[k] /= 2 * (N - 1) - k + 1
#cc_integral = correlate(x,y)
#cc_timeaxis = np.linspace(-(N - 1), N - 1, 2 * N - 1) / fs
#
## Calculate cross correlation using fft method
#cc_fft = fftconvolve(x, y[::-1]) # Reverse y to compute correlation instead of convolve
#
## Plot results
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111) # for some reason this is preferred over fig.gca()
#ax1.set_xlim([-100e-6, 100e-6])
#ax1.set_xlabel('Time delay')
#ax1.set_ylabel('Cross correlation')
#ax1.set_title('Integral method')
#ax1.plot(cc_timeaxis, cc_integral)
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.set_xlim([-100e-6, 100e-6])
#ax2.set_xlabel('Time delay')
#ax2.set_ylabel('Cross correlation')
#ax2.set_title('FFT method')
#ax2.plot(cc_timeaxis, cc_fft)