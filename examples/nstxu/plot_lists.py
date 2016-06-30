# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:05:55 2016

@author: drsmith
"""

import fdp

if __name__=='__main__':
    nstx=fdp.nstxu()
    bes = nstx.s204620.bes
    print(bes.listMethods())
    print(bes.listSignals())
    print(bes.listAttributes())
    sig=bes.d1ch01
    print(sig.listMethods())
    print(sig.listAttributes())
