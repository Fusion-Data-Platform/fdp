# -*- coding: utf-8 -*-
"""
Raw time-series data from BES detectors.

BES detectors generate differential signals such that "zero signal" or
"no light" corresponds to about -9.5 V.  DC signal levels should be referenced
to the "zero signal" output.  "No light" shots (due to failed shutters)
include 138545 and 138858.

BES detector channels **do not** correspond to permanent measurement locations.
BES sightlines observe fixed measurement locations, but sightline optical
fibers can be coupled into any detector channel based upon experimental needs.
Consequently, the measurement location of detector channels can change day to
day.  That said, **most** BES data from 2010 adhered to a standard
configuration with channels 1-8 spanning the radial range R = 129-146 cm.
"""

from .gui import gui
from .fft import fft, plotfft, powerspectrum
from .animation import animate
from .configuration import loadConfig
from .crosspower import plotcrosspower, plotcrossphase, plotcoherence
from .crosspower import crosssignal, plotcorrelation

__all__ = ['fft', 'plotfft', 'powerspectrum',
           'animate', 'loadConfig',
           'gui',
           'crosssignal','plotcrosspower', 'plotcoherence', 'plotcrossphase',
           'plotcorrelation']
