# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:10:56 2016

@author: drsmith
"""

from  ..classes import utilities

def isSignal(obj):
    return utilities.isSignal(obj)

def isContainer(obj):
    return utilities.isContainer(obj)

def isAxis(obj):
    return utilities.isAxis(obj)

def info(obj, *args, **kwargs):
    if utilities.isSignal(obj):
        infoSignal(obj, *args, **kwargs)
    elif utilities.isContainer(obj) or utilities.isShot(obj):
        infoContainer(obj, *args, **kwargs)
    return


def infoSignal(obj, verbose=False, *args, **kwargs):
    obj[:]
    if obj.size == 0:
        return
    print('Name:  {}'.format(dottedPath(obj)))
    if verbose:
        print('  Shot:  {}'.format(obj.shot))
        print('  Description:  {}'.format(obj._desc))
        print('  Title:  {}'.format(obj._title))
        print('  MDS node:  {}'.format(obj._mdsnode))
        print('  MDS tree:  {}'.format(obj._mdstree))
        print('  Shape:  {}'.format(obj.shape))
        for attrname in obj.listAttributes():
            attr = getattr(obj, attrname)
            if utilities.isAxis(attr):
                print('    Axis {}:  {} points'.format(attr._name, attr.size))
            else:
                print('  {}:  {}'.format(attrname, attr))


def infoContainer(obj, *args, **kwargs):
    signalnames = obj.listSignals()
    for signalname in signalnames:
        signal = getattr(obj, signalname)
        signal.info(*args, **kwargs)
    containernames = obj.listContainers()
    for containername in containernames:
        container = getattr(obj, containername)
        container.info(*args, **kwargs)


def dottedPath(obj):
    path = [obj._name]
    current_obj = obj
    while hasattr(current_obj, '_parent'):
        if current_obj._parent is None:
            break
        path.append(current_obj._parent._name)
        current_obj = current_obj._parent
    path.reverse()
    dotted_path = '.'.join(path)
    return dotted_path
