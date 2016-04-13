# -*- coding: utf-8 -*-

import fdp

def fft(signal, power2=None):
    t = signal.time
    nt = t.size
    print(nt)