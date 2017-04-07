# -*- coding: utf-8 -*-
"""
Package-level attributes, methods, and FdfError class

Created on Thu Jun 18 11:18:16 2015

@author: ktritz
"""
import os

VERBOSE = False
TKROOT = None
FDP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FdpError(Exception):
    """
    Error class for FDF package

    **Usage**::

        raise FdfError('error message')

    """

    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message


class FdpWarning(Warning):
    """
    Warning class for FDF package

    **Usage**::

        warnings.warn("message", FdpWarning)

    """

    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message
