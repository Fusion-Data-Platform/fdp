import numpy as np

def shift_dc_signal(obj, data):
    if obj.isSignal() and obj._parent._name=='slow':
        data = data - np.mean(data[0:100])
    return data