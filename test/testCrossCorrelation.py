# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:00:36 2016

@author: dkriete
"""
import numpy as np
from scipy.signal import correlate, fftconvolve, hilbert
import matplotlib.pyplot as plt

# Take correlation between two 10 kHz sine waves with 5 us offset
fs = 2e6 # sampling frequency
T = 10e-3 # 10 ms sampling period
t = np.arange(int(fs * T)) / fs # array of time points
numpnts = len(t) # Number of points in signals (N=800 for this case)

f = 10e3 # sine wave frequency
dt = np.random.normal(5e-6, 10e-6, numpnts) # time offset
x = np.sin(2 * np.pi * f * t)
y = np.sin(2 * np.pi * f * (t + dt))

nperseg = 2000

# =============================================================================
# Begin code copied from CrossSignal class
signal1_seg = []
signal2_seg = []
nseg = numpnts // nperseg # Number of segments
for i in range(nseg):
    start_i = i * nperseg
    signal1_seg.append(x[start_i:start_i + nperseg])
    signal2_seg.append(y[start_i:start_i + nperseg])
signal1_seg = np.array(signal1_seg)
signal2_seg = np.array(signal2_seg)

xcorr = np.zeros((nseg, 2 * nperseg - 1))
autocorr1 = np.zeros((nseg, 2 * nperseg - 1))
autocorr2 = np.zeros((nseg, 2 * nperseg - 1))
xcorr_coef = np.zeros((nseg, 2 * nperseg - 1))
for i in range(nseg):
    
    # Subtract mean from each segment
    signal1_seg[i,:] -= np.mean(signal1_seg, axis=1)[i]
    signal2_seg[i,:] -= np.mean(signal2_seg, axis=1)[i]
    
    # Calculate cross-correlation for each segment
    #     The second input is reversed to change the convolution to a
    #     cross-correlation of the form Sum[x[i] * y[i - k]]. The
    #     output is then reversed to put the cross-correlation into 
    #     the more standard form Sum[x[i] * y[i +k]]
    xcorr[i,:] = fftconvolve(signal1_seg[i,:],
                             signal2_seg[i,::-1])[::-1]
    
    # Calculate autocorrelations for each segment
    autocorr1[i,:] = fftconvolve(signal1_seg[i,:],
                                 signal1_seg[i,::-1])[::-1]
    autocorr2[i,:] = fftconvolve(signal2_seg[i,:],
                                 signal2_seg[i,::-1])[::-1]
    
    # Calculate correlation coefficient
    xcorr_coef[i,:] = xcorr[i,:] / np.sqrt(
            autocorr1[i,nperseg-1] * autocorr2[i,nperseg-1])
    
    # Average over all segments
    crosscorrelation = np.mean(xcorr, axis=0)
    autocorrelation1 = np.mean(autocorr1, axis=0)
    autocorrelation2 = np.mean(autocorr2, axis=0)
    correlation_coef = np.mean(xcorr_coef, axis=0)
    
    # Calculate envelope of correlation using analytic signal method
    correlation_coef_envelope = np.absolute(
            hilbert(correlation_coef))
    
    # Construct time axis for cross correlation
    time_delays = np.linspace(-(nperseg - 1),
                                    (nperseg - 1),
                                   2*nperseg - 1) / fs

fig2 = plt.figure()
ax2 = fig2.add_subplot(111) # for some reason this is preferred over fig.gca()
ax2.set_xlim([0,0.5e-3])
ax2.set_xlabel('Time')
ax2.set_ylabel('y')
ax2.plot(t, y)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111) # for some reason this is preferred over fig.gca()
ax1.set_xlim([-10e-6, 10e-6])
ax1.set_ylim([0.9, 1.0])
ax1.set_xlabel('Time delay')
ax1.set_ylabel('Cross correlation')
ax1.set_title('CrossSignal method')
ax1.plot(time_delays, correlation_coef)

# End code copied from CrossSignal class
# =============================================================================

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