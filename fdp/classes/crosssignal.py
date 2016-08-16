# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:39:37 2016

@author: dkriete
"""

from __future__ import division
from scipy.signal.spectral import _spectral_helper
import numpy as np
from .fdp_globals import FdpError


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

    Attributes
    ==========
    """

    def __init__(self, signal1, signal2, tmin=0.2, tmax=1.0, window='hann',
                 nperseg=None, forcepower2=False, offsetminimum=False,
                 offsetdc=False, normalizetodc=False, degrees=True):

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

        # offsetdc cannot be used with offsetminimum and normalizetodc
        if offsetdc:
            self.detrend = 'constant'
            offsetminimum = False
            normalizetodc = False
        else:
            self.detrend = False

        # Preprocessing of input signals
        self.load_signal()
        if offsetminimum:
            self.apply_offset_minimum()
        self.make_data_window()

        # Calculate spectral quantities
        self.calc_csd()
        self.calc_crosspower()
        self.calc_crossphase()
        self.calc_coherence()
        self.calc_variances2()
        if normalizetodc:
            self.apply_normalize_to_dc()

        # Calculate correlations
        self.calc_correlation()
#        self.calc_correlationcoef()

    def load_signal(self):
        self.signal1[:]
        self.signal2[:]
        self.signal1time[:]
        self.signal2time[:]

    def apply_offset_minimum(self):
        'Shift signal so that first 10,000 points are near zero'
        zerolevel1 = np.min(self.signal1[:1e4])
        self.signal1 -= zerolevel1
        zerolevel2 = np.min(self.signal2[:1e4])
        self.signal2 -= zerolevel2

    def make_data_window(self):
        'Reduce signals to only contain data in the specified time window'
        mask1 = np.logical_and(self.signal1time >= self.tmin,
                               self.signal1time <= self.tmax)
        mask2 = np.logical_and(self.signal2time >= self.tmin,
                               self.signal2time <= self.tmax)
        self.signal1 = np.extract(mask1, self.signal1)
        self.signal2 = np.extract(mask2, self.signal2)
        self.signal1time = np.extract(mask1, self.signal1time)
        self.signal2time = np.extract(mask2, self.signal2time)

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
        """

        # Calculate the sampling rate. Signal1 and signal2 must have the same 
        # sampling rate.
        fs = 1 / np.mean(np.diff(self.signal1time[:1e4]))
        
        # If the number of points per segement is not specified, calculate the
        # number that gives approximately equal time and frequency resolution.
        if self.nperseg is None:
            self.nperseg = np.int(np.sqrt(2*len(self.signal1)))
        
        # Use next power of 2 for nperseg if specified. FFT algorithm is most 
        # efficient when nperseg is a power of 2.
        if self.forcepower2 is True:
            self.nperseg = np.power(2, int(np.log2(self.nperseg-1))+1)
            
        # Result of csd calculates is a 2D array. Axis 0 is the frequency axis
        # and axis 1 is the time axis. Entries in the times array are the 
        # center values for each time bin
            
        # Calculate cross spectral density
        self.freqs, self.times, self.csd = _spectral_helper(
                self.signal1,
                self.signal2,
                fs=fs,
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
                fs=fs,
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
                fs=fs,
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
        
        # Convert to degrees if specified
        if self.degrees:
            self.crossphase = np.rad2deg(self.crossphase)
            self.crossphase_binavg = np.rad2deg(self.crossphase_binavg)
        
        
    def calc_coherence(self):
        'Calculate the coherence'
        # Note that the coherence in each time bin is identically 1 so there
        # is no use in an unaveraged coherence.
        self.coherence = np.sqrt(np.absolute(self.csd_binavg)**2 
                        / (self.asd1_binavg * self.asd2_binavg))
        
        # Calculate the minimum statistically significant coherence under a 
        # 95% confidence interval
        if np.shape(self.csd)[-1] == 1:
            self.minsigcoh = 1
        else:
            self.minsigcoh = np.sqrt(1 - 0.05**(1/(np.shape(self.csd)[-1] - 1)))

    def calc_variances(self):
        'Calculate the variance of the crosspower, crossphase, and coherence'
        # The equations below give standard deviations larger than the mean
        # for the bulk of the data. This doesn't seem correct so they should
        # not be used for now (8/10/2016)
        
        # Start by calculating the mean, variance, and covariance for the real
        # and imaginary parts of the cross spectral density at each frequency.
        realmean = np.mean(np.real(self.csd), axis=-1)
        imagmean = np.mean(np.imag(self.csd), axis=-1)
        realvar = np.var(np.real(self.csd), axis=-1, ddof=1)
        imagvar = np.var(np.imag(self.csd), axis=-1, ddof=1)
        
        covar = np.empty(np.shape(self.csd)[0])
        for i in range(np.shape(self.csd)[0]):
            # For each frequency, split the 1D complex array of data for each 
            # time bin into a 2D array where first row is real part and second
            # row is imaginary part. 
            reform = np.array([np.real(self.csd[i,:]),np.imag(self.csd[i,:])])
            # Pull the covariance between real and imaginary parts out of the
            # covariance array.
            cov = np.cov(reform)
            covar[i] = cov[0,1]
            
            # Diagonal terms of covariance matrix should be variances
            if not np.allclose(realvar, cov[0,0]):
                raise FdpError('Real variance mismatch')
            if not np.allclose(imagvar, cov[1,1]):
                raise FdpError('Imaginary variance mismatch')
            # Covariance matrix should be symmetric
            if not np.allclose(cov[1,0], cov[0,1]):
                raise FdpError('Covariance mismatch')
        
        # The following variance formulae are derived by applying the law of 
        # propagation of uncertainty to the equations used to calculate 
        # crosspower, crossphase, and coherence. The formulae are for the
        # general case where the real and imaginary parts of the cross spectral
        # density may be correlated.
        
        # Calculate the variance of the crosspower
        self.crosspower_var = ((realvar*realmean**2 + imagvar*imagmean**2
                            + 2*realmean*imagmean*covar)
                            / (realmean**2 + imagmean**2))
        
        # Calculate the variance of the crossphase
        self.crossphase_var = ((realvar*imagmean**2 + imagvar*realmean**2
                            - 2*realmean*imagmean*covar)
                            / ((realmean**2 + imagmean**2)**2))
        
#        self.coherence_var = 
    
    def calc_variances2(self):
        'Calculate variance of crosspower and crossphase'
        # This method calculates the variance of the crosspower by taking the 
        # magnitude of each cross spectral density time bin and then computing
        # the variance of these magnitudes. This is not a statistically correct
        # way of calculating the variance but the other method I was trying 
        # (in calc_variances) didn't seem to work so this is an alternative.
        
        self.crosspower_var = np.var(self.crosspower, axis=-1, ddof=1)
        self.crossphase_var = np.var(self.crossphase, axis=-1, ddof=1)
        
    def apply_normalize_to_dc(self):
        'Normalize by dividing by the zero frequency value'
        for i in range(np.shape(self.crosspower)[-1]):
            self.asd1[:,i] /= self.asd1[0,i]
            self.asd2[:,i] /= self.asd2[0,i]
            self.crosspower[:,i] /= self.crosspower[0,i]
        
        #Normalize the crosspower variance
        self.crosspower_var /= (self.crosspower_binavg[0]**2)
        
        self.asd1_binavg /= self.asd1_binavg[0]
        self.asd2_binavg /= self.asd2_binavg[0]
        self.crosspower_binavg /= self.crosspower_binavg[0]
        
    def calc_correlation(self):
        'Calculate cross correlation of the fluctuating parts of input signals'
        # Requires both signals to have the same time basis
        if len(self.signal1) == len(self.signal2):
            
            # Calculate cross correlation using Numpy method
            self.correlation = np.correlate(
                    self.signal1 - np.mean(self.signal1),
                    self.signal2 - np.mean(self.signal2),
                    mode='Full')
            self.correlation /= len(self.signal1)
                    
            # Normalize correlation to produce correlation coefficient
            self.correlation_coef = self.correlation / np.sqrt(
                    np.var(self.signal1) * np.var(self.signal2))
                                            
            # Construct time axis for cross correlation
            delta_t = np.mean(np.diff(self.signal1time[:1e4]))
            n = len(self.signal1)
            self.time_delays = delta_t * np.linspace(-(n-1), (n-1), 2*n-1)
        else:
            raise FdpError('Input signals are different lengths')
        
#    def calc_correlationcoef(self):