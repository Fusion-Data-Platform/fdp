# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:39:37 2016

@author: dkriete
"""

from __future__ import division
from scipy.signal import firwin, filtfilt, fftconvolve, hilbert
from scipy.signal.spectral import _spectral_helper
import numpy as np
from .fdp_globals import FdpError
import time

class CrossSignal(object):
    """
    CrossSignal class

    Calculates spectral quantities which take 2 signals as inputs

    Parameters
    ==========
    signal1 : fdp signal
        First signal to be analyzed.
    signal2 : fdp signal
        Second signal to be analyzed. Both signals must be uniformly sampled
        at the same frequency.
    tmin : float, optional
        Start time of the time window to be analyzed. Defaults to 0.2.
    tmax : float, optional
        End time of the time window to be analyzed. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Window to be applied to each data segment. See scipy.signal.get_window
        documentation for full list of windows available. If an array is given
        it will be used as the window. Defaults to hann window.
    nperseg : int, optional
        Number of points in each segment. More points per segment increases
        frequency resolution but also increases variance (noisiness) of the
        spectral density estimate. Defaults to a value which roughly evenly
        trades off between frequency resolution and variance.
    forcepower2 : bool, optional
        If True, force the number of points per segment to be the next largest
        power of 2. This increases efficiency of FFT calculation. Defaults to
        False.
    offsetminimum : bool, optional
        Offsets the signals so that the beginning of the data is near the zero
        level. Useful for diagnostics where zero input signal leads to nonzero
        output signal. Defaults to False.
    offsetdc : bool, optional
        Offsets each segment of the input signals so that they have zero mean.
        Cannot be used with offsetminimum or normalizetodc. Defaults to False.
    normalizetodc : bool, optional
        Normalizes the cross spectral density and cross power by their dc
        level. Defaults to False.
    degrees : bool, optional
        If True, the cross phase output will be in degrees. If False, the cross
        phase output will be in radians. Defaults to True.
    fmin : float, optional
        Lower cutoff frequency in kHz for band pass filter applied to data.
        Defaults to 0.
    fmax : float, optional
        Higher cutoff frequency in kHz for band pass filter applied to data.
        Defaults to Nyquist frequency.
    """

    def __init__(self, signal1, signal2, tmin=0.2, tmax=1.0, window='hann',
                 nperseg=None, forcepower2=False, offsetminimum=False,
                 offsetdc=False, normalizetodc=False, degrees=True,
                 fmin=None, fmax=None):

        self.signal1 = signal1
        self.signal2 = signal2
        self.signal1time = signal1.time
        self.signal2time = signal2.time
        self.signal1name = signal1._name
        self.signal2name = signal2._name
        self.parent1name = signal1._parent._name
        self.parent2name = signal2._parent._name
        self.shot = signal1.shot

        self.tmin = tmin
        self.tmax = tmax
        self.window = window
        self.nperseg = nperseg
        self.forcepower2 = forcepower2
        self.degrees = degrees
        self.fmin = fmin
        self.fmax = fmax

        # offsetdc cannot be used with offsetminimum and normalizetodc
        t0 = time.time()
        if offsetdc:
            self.detrend = 'constant'
            offsetminimum = False
            normalizetodc = False
        else:
            self.detrend = False
        print 'Time to offset/detrend data: ' + str(time.time()-t0)

        # Preprocessing of input signals
        self.load_signals()
        if offsetminimum:
            self.apply_offset_minimum()
        print 'Time to load/offset data: ' + str(time.time()-t0)
        self.make_data_window()
        print 'Time to make data windows: ' + str(time.time()-t0)
        self.filter_signals()
        print 'Time to filter data: ' + str(time.time()-t0)

        # Calculate spectral quantities
        self.calc_csd()
        print 'Time to calculate spectral densities: ' + str(time.time()-t0)
        self.calc_crosspower()
        print 'Time to calculate crosspower: ' + str(time.time()-t0)
        self.calc_crossphase()
        print 'Time to calculate crossphase: ' + str(time.time()-t0)
        self.calc_coherence()
        print 'Time to calculate coherence: ' + str(time.time()-t0)
        self.calc_error()
        print 'Time to calculate error: ' + str(time.time()-t0)
        if normalizetodc:
            self.apply_normalize_to_dc()
        print 'Time to normalize data: ' + str(time.time()-t0)

        # Calculate correlations
        self.calc_correlation_fft()
        print 'Time to calculate correlation coefficient: ' + str(time.time()-t0)

    def load_signals(self):
        """
        Load data and check to ensure each signal has same time scaling.
        """
        
        # Load data
        self.signal1[:]
        self.signal2[:]
        self.signal1time[:]
        self.signal2time[:]
        
        # Check to ensure both signals have same sampling rate
        fs1 = 1 / np.mean(np.diff(np.array(self.signal1time))) # Not sure if np.array() needed
        fs2 = 1 / np.mean(np.diff(np.array(self.signal2time)))
        if abs(fs1 - fs2) < 1e-3:
            self.fSample = (fs1 + fs2) / 2
            self.fNyquist = self.fSample / 2
        else:
            raise FdpError('Input signals have different sampling rates')

    def apply_offset_minimum(self):
        'Shift signal so that first 1,000 points are near zero'
        
        zerolevel1 = np.mean(self.signal1[:1e3])
        self.signal1 -= zerolevel1
        zerolevel2 = np.mean(self.signal2[:1e3])
        self.signal2 -= zerolevel2

    def make_data_window(self):
        """
        Reduce signals to only contain data in the specified time window. Then
        check to ensure each signal has same length.
        """
        
        mask1 = np.logical_and(self.signal1time >= self.tmin,
                               self.signal1time <= self.tmax)
        mask2 = np.logical_and(self.signal2time >= self.tmin,
                               self.signal2time <= self.tmax)
        self.signal1 = np.extract(mask1, self.signal1)
        self.signal2 = np.extract(mask2, self.signal2)
        self.signal1time = np.extract(mask1, self.signal1time)
        self.signal2time = np.extract(mask2, self.signal2time)
        
        # Check to ensure both signals have same number of points
        if len(self.signal1) == len(self.signal2):
            self.numpnts = len(self.signal1)
        else:
            raise FdpError('Input signals are different lengths')
    
    def filter_signals(self):
        'Band pass filter the input data'
        
        # Check to see if either filter frequency has been set. If not, then
        # don't filter the data
        if self.fmin is not None or self.fmax is not None:
            
            # Set default values for unspecified frequencies and convert units
            # of specified frequencies from kHz to Hz
            if self.fmin is None:
                self.fmin = 0.
            else:
                self.fmin *= 1000
            if self.fmax is None:
                self.fmax = self.fNyquist
            else:
                self.fmax *= 1000
            
            # Verify that frequencies are valid
            if self.fmin < 0 or self.fmin > self.fNyquist:
                raise FdpError('fmin is outside valid range')
            if self.fmax < 0 or self.fmax > self.fNyquist:
                raise FdpError('fmax is outside valid range')
            if self.fmax < self.fmin:
                raise FdpError('fmin is larger than fmax')
            
            # Filter data using 501 tap FIR filter generated using window
            # method (Hamming window)
            desirednumtaps = 501
            if self.numpnts <= desirednumtaps:
                numtaps = 2 * (self.numpnts // 2) - 1
            else:
                numtaps = desirednumtaps
            
            h = firwin(numtaps, [self.fmin, self.fmax], pass_zero=False,
                       nyq=self.fNyquist)
            self.signal1 = filtfilt(h, 1.0, self.signal1, padlen=numtaps)
            self.signal2 = filtfilt(h, 1.0, self.signal2, padlen=numtaps)

    def calc_csd(self):
        """
        Calculate the cross spectral density using Scipy csd function.

        csd utilizes Welch's method to estimate spectral density. Data is
        split into overlapping segments. Each segment is windowed, then the 
        cross spectral density is calculated using Fourier transforms. The 
        results from all windows are averaged together to produce a lower
        variance estimate of the spectral density.

        A segment overlap factor of 2 is used (50% overlap).
        A one-sided spectrum is returned for real inputs
        The cross spectral density (units V**2/Hz) is calculated, as
        opposed to the cross spectrum (units V**2).
        
        csd is a 2D array containing the cross spectral density. Axis 0 is the
        frequency axis and axis 1 is the time axis. Entries in the times array
        are the center values for each time bin.
        """
        
        # If the number of points per segement is not specified, calculate the
        # number that gives approximately equal time and frequency resolution.
        if self.nperseg is None:
            self.nperseg = int(np.sqrt(2 * self.numpnts))
        
        # Use next power of 2 for nperseg if specified. FFT algorithm is most 
        # efficient when nperseg is a power of 2.
        if self.forcepower2 is True:
            self.nperseg = np.power(2, int(np.log2(self.nperseg - 1)) + 1)
        
        # Calculate cross spectral density
        self.freqs, self.times, self.csd = _spectral_helper(
                self.signal1,
                self.signal2,
                fs=self.fSample,
                window=self.window,
                nperseg=self.nperseg,
                detrend=self.detrend,
                scaling='density',
                mode='psd'
            )
        
        # Calculate auto spectral density of signal 1
        _, _, self.asd1 = _spectral_helper(
                self.signal1,
                self.signal1,
                fs=self.fSample,
                window=self.window,
                nperseg=self.nperseg,
                detrend=self.detrend,
                scaling='density',
                mode='psd'
            )
        
        # Calculate auto spectral density of signal 2
        _, _, self.asd2 = _spectral_helper(
                self.signal2,
                self.signal2,
                fs=self.fSample,
                window=self.window,
                nperseg=self.nperseg,
                detrend=self.detrend,
                scaling='density',
                mode='psd'
            )
        
        # Calculate time bin averaged spectral densities
        self.csd_binavg = np.mean(self.csd, axis=-1)
        self.asd1_binavg = np.mean(self.asd1, axis=-1)
        self.asd2_binavg = np.mean(self.asd2, axis=-1)
        
        # Record number of bins (aka # segments or # realizations) in the ffts
        self.numbins = np.shape(self.csd)[-1]
        
        # Convert frequency units from Hz to kHz
        self.freqs /= 1000
        
        # Shift time bins to correspond to original data window
        self.times += (self.signal1time[0] + self.signal2time[0]) / 2
    
    def calc_crosspower(self):
        'Calculate the cross power (magnitude of cross spectral density)'
        
        self.crosspower = np.absolute(self.csd)
        self.crosspower_binavg = np.absolute(self.csd_binavg)
        
    def calc_crossphase(self):
        """
        Calculate the cross phase (phase angle of cross spectral density)
        Result is between -180 degrees and 180 degrees (or -pi/2 to pi/2)
        """
        
        self.crossphase = np.angle(self.csd)
        self.crossphase_binavg = np.angle(self.csd_binavg)
        
        # Unwrap phase to get rid of jumps from +pi to -pi (or vice-versa)
        # Doesn't seem to be fixing all the phase jumps
        self.crossphase = np.unwrap(self.crossphase, axis=0)
        self.crossphase_binavg = np.unwrap(self.crossphase_binavg)
        
        # Convert to degrees if requested
        if self.degrees:
            self.crossphase = np.rad2deg(self.crossphase)
            self.crossphase_binavg = np.rad2deg(self.crossphase_binavg)
        
    def calc_coherence(self):
        'Calculate the magnitude squared coherence'
        
        # Calculate magnitude squared coherence
        # Note that the coherence in each time bin is identically 1 so there
        # is no use in an unaveraged coherence.
        self.mscoherence = (np.absolute(self.csd_binavg)**2 
                       / (self.asd1_binavg * self.asd2_binavg))
        
        # Calculate coherence (square root of magnitude squared coherence)
        self.coherence = np.sqrt(self.mscoherence)
        
        # Calculate the minimum statistically significant coherence for a 
        # 95% confidence interval
        if self.numbins == 1:
            self.minsig_mscoherence = 1
            self.minsig_coherence = 1
        else:
            self.minsig_mscoherence = 1 - 0.05**(1/(self.numbins - 1))
            self.minsig_coherence = np.sqrt(self.minsig_mscoherence)
    
    def calc_error(self):
        """
        Calculate random error of coherence and crossphase using 
        formulae from Bendat & Piersol textbook.
        """
        
        # Bendat & Piersol eq 11.62
        self.crossphase_error = (np.sqrt(1 - self.coherence)
                              / (self.mscoherence * np.sqrt(2 * self.numbins)))
        if self.degrees:
            self.crossphase_error = np.rad2deg(self.crossphase_error)
        
        # Bendat & Piersol eq 11.47 (multiplied by coherence to get std dev)
        self.mscoherence_error = (np.sqrt(2 * self.coherence / self.numbins)
                             * (1 - self.coherence))
                             
        # Use of law of propagation of uncertainty to calculate coherence error
        self.coherence_error = self.mscoherence_error / (2 * self.coherence)
        
    def apply_normalize_to_dc(self):
        'Normalize by dividing by the zero frequency value'
        
        for i in range(self.numbins):
            self.asd1[:,i] /= self.asd1[0,i]
            self.asd2[:,i] /= self.asd2[0,i]
            self.crosspower[:,i] /= self.crosspower[0,i]
        
        self.asd1_binavg /= self.asd1_binavg[0]
        self.asd2_binavg /= self.asd2_binavg[0]
        self.crosspower_binavg /= self.crosspower_binavg[0]
        
    def calc_correlation_fft(self):
        """
        Calculate cross correlation of fluctuation component of input signals 
        using the fft method to perform the convolution. This is faster than 
        the integral definition method.
        
        Returns R(tau), where R(-tau) = F^-1[F[x(t)] * F[y(-t)]] and F[]
        denotes the Fourier transform. This calculation is equivalent to 
        R[k] = Sum_i[x[i] * y[i + k]]
        """
        # Segment the signals with nperseg points in each segment
        signal1_seg = []
        signal2_seg = []
        nseg = self.numpnts // self.nperseg # Number of segments
        for i in range(nseg):
            start_i = i * self.nperseg
            signal1_seg.append(self.signal1[start_i:start_i + self.nperseg])
            signal2_seg.append(self.signal2[start_i:start_i + self.nperseg])
        signal1_seg = np.array(signal1_seg)
        signal2_seg = np.array(signal2_seg)
        
        xcorr = np.zeros((nseg, 2 * self.nperseg - 1))
        autocorr1 = np.zeros((nseg, 2 * self.nperseg - 1))
        autocorr2 = np.zeros((nseg, 2 * self.nperseg - 1))
        xcorr_coef = np.zeros((nseg, 2 * self.nperseg - 1))
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
                    autocorr1[i,self.nperseg-1] * autocorr2[i,self.nperseg-1])
            
            # Average over all segments
            self.crosscorrelation = np.mean(xcorr, axis=0)
            self.autocorrelation1 = np.mean(autocorr1, axis=0)
            self.autocorrelation2 = np.mean(autocorr2, axis=0)
            self.correlation_coef = np.mean(xcorr_coef, axis=0)
            
            # Calculate envelope of correlation using analytic signal method
            self.correlation_coef_envelope = np.absolute(
                    hilbert(self.correlation_coef))
            
            # Construct time axis for cross correlation
            self.time_delays = np.linspace(-(self.nperseg - 1),
                                            (self.nperseg - 1),
                                           2*self.nperseg - 1) / self.fSample
        
    def calc_correlation(self):
        """
        Calculate cross correlation of the fluctuating parts of input signals. 
        Warning: this method is slow, calc_correlation_fft is a faster 
        alternative.s
        """
        
        # Calculate cross correlation using Numpy method
        self.crosscorrelation = np.correlate(
                self.signal1 - np.mean(self.signal1),
                self.signal2 - np.mean(self.signal2),
                mode='Full')
        self.crosscorrelation /= self.numpnts
        
        # Normalize correlation to produce correlation coefficient
        self.correlation_coef = self.crosscorrelation / np.sqrt(
                np.var(self.signal1) * np.var(self.signal2))
        
        # Calculate envelope of correlation using analytic signal method
        self.correlation_coef_envelope = np.absolute(
                hilbert(self.correlation_coef))
        
        # Construct time axis for cross correlation
        delta_t = 1 / self.fSample
        self.time_delays = delta_t * np.linspace(-(self.numpnts - 1),
                                                  (self.numpnts - 1), 
                                                 2*self.numpnts - 1)