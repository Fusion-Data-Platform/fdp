# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 21:00:08 2016

@author: drsmith
"""

import numpy as np
from scipy import fftpack
from .fdp_globals import FdpError

class Fft(object):
    """
    Fft class
    
    Calculates binned ffts for time interval tmin to tmax.
    
    Attributes
        fft: complex-valued fft(time, freq),
            single-sided for real input, dboule-sided for complex input
        time: time array for bins [s]
        freq: frequency array [kHz]
        power2: # of points in fft, power-of-2 (>=) enforced
    """
    
    def __init__(self, signal, power2=None, tmin=0.2, tmax=1.0, 
                 hanning=True, offsetminimum=False, offsetdc=False,
                 normalizetodc=False):
        self.signal = signal
        self.signalname = signal._name
        self.parentname = signal._parent._name
        self.shot = signal.shot
        self.power2 = power2 # in None, guess value below
        self.hanning = hanning # true for Hann window
        self.offsetminimum = offsetminimum # true to shift signal by minimum
        self.offsetdc = offsetdc # true to remove DC component
        self.normalizetodc = normalizetodc
        
        if self.hanning:
            # 50% overlap for Hanning window
            self.overlapfactor = 2
        else:
            # no overlap among bins
            self.overlapfactor = 1
        
        if tmax>10:
            # assume ms input and convert to s
            tmin = tmin/1e3
            tmax = tmax/1e3
        self.tmin = tmin
        self.tmax = tmax
        
        # single-sided, complex-valued fft(time, freq)
        self.fft = None
        # frequency array
        self.freq = None
        # array of center times for bins
        self.time = None
        
        self.nbins = None
        self.window = None
        self.iscomplexsignal = None
        
        # real, positive definite power spec. density, psd(freq,time)
        self.psd = None
        # input signal integrated power, intpower(time)
        self.intpower = None
        self.maxintpowererror = None
        
        self.loadSignal()
        self.makeTimeBins()
        if self.offsetminimum:
            self.applyMinimumOffset()
        if self.offsetdc:
            self.normalizetodc = False
            self.applyDcOffset()
        if self.hanning:
            self.applyHanningWindow()
        self.calcIntegratedSignalPower()
        self.calcFft()
        if self.normalizetodc and not self.offsetdc:
            self.applyNormalizeToDc()
        self.calcPsd()
        
    def loadSignal(self):
        self.signal[:]
        self.signal.time[:]
        # real-valued floating or complex-valued?
        if self.signal.dtype.kind == 'f':
            self.iscomplexsignal = False
        elif self.signal.dtype.kind == 'c':
            self.iscomplexsignal = True
        else:
            raise FdpError('Data must be floating or complex')
            
        
    def makeTimeBins(self):
        self.time = []
        self.fft = []
        time_indices = np.where(np.logical_and(self.signal.time>=self.tmin,
                                                self.signal.time<=self.tmax))[0]
        istart = time_indices[0]
        istop = time_indices[time_indices.size-1]
        if self.power2 is None:
            # guess appropriate pwoer2 value
            self.power2 = np.int(np.sqrt((istop-istart+1)*self.overlapfactor))
        self.power2 = nextpow2(self.power2)
        t = np.mean(self.signal.time[istart:istart+self.power2])
        while t.item(0)<=self.tmax:
            self.time.append(t)
            self.fft.append(self.signal[istart:istart+self.power2])
            # candidate istart and t for next iteration
            istart = istart + self.power2/self.overlapfactor
            t = np.mean(self.signal.time[istart:istart+self.power2])
        # convert lists to ndarrays
        # at this point, fft contains modified input signals
        self.fft = np.array(self.fft)
        self.time = np.array(self.time)
        self.nbins = self.time.size

    def applyMinimumOffset(self):
        zerosignal = np.min(self.signal[0:1e4])
        self.fft -= zerosignal
    
    def applyDcOffset(self):
        # remove DC offset bin-wise
        for i in range(self.nbins):
            self.fft[i,:] -= np.mean(self.fft[i,:])
    
    def applyHanningWindow(self):
        self.window = np.hanning(self.power2)
        for i in range(self.nbins):
            self.fft[i,:] = np.multiply(self.fft[i,:], self.window)
    
    def calcIntegratedSignalPower(self):
        self.intpower = np.sum(np.square(self.fft), axis=1)
    
    def calcFft(self):
        timeint = np.mean(np.diff(self.signal.time[0:1e4]))
        # complex-valued, double-sided FFT
        self.fft = fftpack.fft(self.fft, 
                               n=self.power2,
                               axis=1)
        # frequency array in kHz
        self.freq = fftpack.fftfreq(self.power2, d=timeint)/1e3
        # check integrated power (bin-wise)
        self.checkIntegratedPower()
        # if real-valued input, convert to single-sided FFT
        if not self.iscomplexsignal:
            # complex-valued, single-sided FFT
            ssfft = self.fft[:,0:self.power2/2+1].copy()
            ssfft[:,1:self.power2/2] *= np.sqrt(2.0)
            self.fft = ssfft
            self.freq = self.freq[0:self.power2/2+1].copy()
            self.freq[self.power2/2] = -self.freq[self.power2/2]
            # check integrated power (bin-wise)
            self.checkIntegratedPower()
            
    def applyNormalizeToDc(self):
        for i in range(self.nbins):
            self.fft[i,:] /= np.real(self.fft[i,0])
            
    def calcPsd(self):
        # PSD in dB: 10*log10 (|FFT|^2)
        self.psd = 10*np.log10(np.square(np.absolute(self.fft)))
                
    def checkIntegratedPower(self):
        intpowercheck = np.sum(np.square(np.absolute(self.fft)),
                               axis=1)/self.power2
        if not np.allclose(self.intpower, intpowercheck):
            raise FdpError('Integrated power mismatch')
        


def nextpow2(number):
    """Return next power of 2 (>= number)"""
    exp = int(np.log2(number-1))+1
    return np.power(2, exp)
