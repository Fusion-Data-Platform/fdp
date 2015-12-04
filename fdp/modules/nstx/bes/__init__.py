# -*- coding: utf-8 -*-
"""
Raw time-series data from BES detectors.

BES detectors generate differential signals such that "zero signal" or "no light" corresponds to about -9.5 V.  DC signal levels should be referenced to the "zero signal" output.  "No light" shots (due to failed shutters) include 138545 and 138858, 

BES detector channels **do not** correspond to permanent measurement locations.  BES sightlines observe fixed measurement locations, but sightline optical fibers can be coupled into any detector channel based upon experimental needs.  Consequently, the measurement location of detector channels can change day to day.  That said, **most** BES data from 2010 adhered to a standard configuration with channels 1-8 spanning the radial range R = 129-146 cm.
"""

description = __doc__


if __name__ == '__main__':
    print(description)
    