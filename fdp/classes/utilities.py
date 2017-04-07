# -*- coding: utf-8 -*-

from . import container, signal, shot


def isContainer(obj):
    return issubclass(type(obj), container.Container) and 'Container' in str(type(obj))


def isSignal(obj):
    return issubclass(type(obj), signal.Signal) and 'Signal' in str(type(obj))


def isAxis(obj):
    return issubclass(type(obj), signal.Signal) and 'Axis' in str(type(obj))


def isShot(obj):
    return issubclass(type(obj), shot.Shot) and 'Shot' in str(type(obj))
