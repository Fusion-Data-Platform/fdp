# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:27:21 2015

@author: ktritz
"""
from builtins import str, object
import inspect
import types

from .parse import parse_mdspath


class Node(object):
    """
    Node class
    """

    def __init__(self, element, parent=None):
        self._parent = parent
        self._name = element.get('name')
        self.mdsnode = parse_mdspath(self, element)[0]
        self._data = None
        self._title = element.get('title')
        self._desc = element.get('desc')
        self.units = element.get('units')

    def __repr__(self):
        if self._data is None:
            self._data = self._get_mdsdata(self)
        return str(self._data)

    def __getattr__(self, attribute):
        if attribute is '_parent':
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        if self._parent is None:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        attr = getattr(self._parent, attribute)
        if inspect.ismethod(attr):
            return types.MethodType(attr.__func__, self)
        else:
            return attr
