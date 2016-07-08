# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:10:56 2016

@author: drsmith
"""

from fdp.classes.utilities import isSignal, isContainer, isAxis, isShot

def info(obj):
    if isSignal(obj):
        infoSignal(obj)
    elif isContainer(obj) or isShot(obj):
        infoContainer(obj)
    return
    
def infoSignal(obj):
    obj[:]
    print('Name:  {}'.format(obj._name))
    print('  Parents:  {}'.format(dottedParents(obj)))
    print('  Shot:  {}'.format(obj.shot))
    print('  Description:  {}'.format(obj._desc))
    print('  Title:  {}'.format(obj._title))
    print('  MDS node:  {}'.format(obj._mdsnode))
    print('  MDS tree:  {}'.format(obj._mdstree))
    print('  Shape:  {}'.format(obj.shape))
    for attrname in obj.listAttributes():
        attr = getattr(obj, attrname)
        if isAxis(attr):
            print('    Axis {}:  {} points'.format(attr._name, attr.size))
        else:
            print('  {}:  {}'.format(attrname, attr))

def infoContainer(obj):
    signalnames = obj.listSignals()
    for signalname in signalnames:
        signal = getattr(obj, signalname)
        signal.info()
    containernames = obj.listContainers()
    for containername in containernames:
        container = getattr(obj, containername)
        container.info()
        
def dottedParents(obj):
    parents = []
    current_obj = obj
    while hasattr(current_obj, '_parent'):
        if current_obj._parent is None:
            break
        parents.append(current_obj._parent._name)
        current_obj = current_obj._parent
    parents.reverse()
    dotted_parents = '.'.join(parents)
    return dotted_parents