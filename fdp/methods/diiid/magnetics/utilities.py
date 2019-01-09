
def current_convert(signal, data):
    if signal.isSignal():
        if signal._name=='ip':
            data = data / 1e6
        if signal._name in ['ecoila' or 'bcoil']:
            data = data / 1e3
    return data