
def power_convert(obj, data):
    if obj.isSignal():
        if obj._name== 'pinj':
            data = data / 1e3
        else:
            data = data / 1e6
    return data