# -*- coding: utf-8 -*-
"""


@author: jschmitt
"""

import numpy as np
import matplotlib.pyplot as plt
import math as math

def plot(signal, overwrite=False):
    print(signal._name)
    
    if signal._name in ['ip']:
        if not overwrite:
            plt.figure()
            plt.subplot(1, 1, 1)
        t = signal.time[:] /1000 # ms to sec
        d = signal.data[:]
        #signal.data[:]
        plt.plot(t, d)
        if not overwrite:
            plt.suptitle('Shot #{}'.format(signal.shot), x=0.5, y=1.00,
                         fontsize=20, horizontalalignment='center')
            plt.title(signal._name, fontsize=20)
            plt.ylabel('{} ({})'.format('Current', signal.data.units))
            plt.xlabel('{} ({})'.format('Time', signal.time.units))
            plt.show()
