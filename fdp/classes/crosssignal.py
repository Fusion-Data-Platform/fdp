# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:39:37 2016

@author: dkriete
"""

from scipy.signal.spectral import _spectral_helper
import numpy as np


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
        self.calc_cohere()
        if normalizetodc:
            self.apply_normalize_to_dc()

        # Calculate correlations
#        self.calc_xcorr()
#        self.calc_xcorrcoef()

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
        self.crossphase = np.angle(self.csd, deg=self.degrees)
        self.crossphase_binavg = np.angle(self.csd_binavg, deg=self.degrees)
        
        # Unwrap phase
        
    def calc_cohere(self):
        'Calculate the magnitude squared coherence'
        self.cohere = np.absolute(self.csd)**2 / (self.asd1 * self.asd2)
        self.cohere_binavg = (np.absolute(self.csd_binavg)**2 
                              / (self.asd1_binavg * self.asd2_binavg))
        
    def apply_normalize_to_dc(self):
        'Normalize by dividing by the zero frequency value'
        # Not sure about normalization of the cross spectral density
        # 1 - does it make sense to normalize csd by 0 frequency value?
        # 2 - should it divide by real part or magnitude of 0 frequency value?
        self.csd /= np.real(self.csd[0,:])
        self.csd_binavg /= np.real(self.csd_binavg[0])
        self.asd1 /= self.asd1[0,:]
        self.asd1_binavg /= self.asd1_binavg[0]
        self.asd2 /= self.asd2[0,:]
        self.asd2_binavg /= self.asd2_binavg[0]
        self.crosspower /= self.crosspower[0,:]
        self.crosspower_binavg /= self.crosspower_binavg[0]
        
#    def calc_xcorr(self):
#        
#    def calc_xcorrcoef(self):