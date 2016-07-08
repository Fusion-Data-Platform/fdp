# -*- coding: utf-8 -*-

from . import container, fdpsignal, shot

def isContainer(obj):
    return issubclass(obj.__class__, container.Container) and 'Container' in repr(type(obj))

def isSignal(obj):
    return issubclass(obj.__class__, fdpsignal.Signal) and 'Signal' in repr(type(obj))

def isAxis(obj):
    return issubclass(obj.__class__, fdpsignal.Signal) and 'Axis' in repr(type(obj))

def isShot(obj):
    return issubclass(obj.__class__, shot.Shot) and 'Shot' in repr(type(obj))