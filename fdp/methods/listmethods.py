# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:18:01 2016

@author: drsmith
"""

from ..lib.utilities import isSignal, isContainer


def listSignals(obj):
    attrnames = dir(obj)
    signals = []
    for attrname in attrnames:
        attr = getattr(obj, attrname)
        if isSignal(attr):
            signals.append(attrname)
    return signals


def listContainers(obj):
    attrnames = dir(obj)
    containers = []
    for attrname in attrnames:
        attr = getattr(obj, attrname)
        if isContainer(attr):
            containers.append(attrname)
    return containers


def listMethods(obj):
    methods = []
    while True:
        attrnames = dir(obj)
        for attrname in attrnames:
            if attrname not in methods:
                attr = getattr(obj, attrname)
                if hasattr(attr, '__func__'):
                    methods.append(attrname)
        if hasattr(obj, '_parent'):
            obj = obj._parent
        else:
            break
    return methods


def listAttributes(obj):
    attrnames = dir(obj)
    attributes = []
    for attrname in attrnames:
        attr = getattr(obj, attrname)
        if not isContainer(attr) and \
                not isSignal(attr) and \
                not hasattr(attr, '__func__'):
            attributes.append(attrname)
    return attributes
