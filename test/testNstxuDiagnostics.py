# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:21:19 2016

@author: drsmith
"""

import unittest
import fdp
from fdp.classes.utilities import isContainer, isSignal, isAxis
from setup import SetupNstxu

print('running tests in {}'.format(__file__))


class TestNstxuDiagnostics(SetupNstxu):

    def testSignalAxes(self, container=None):
        """
        Recursively parse tree
        Assert all signals contain 'axes' attribute
        Assert all signals contain 'time' attribute
        Assert all 'axes' elements are axis objects
        Assert all axis attributes are in 'axes'
        """
        if not container:
            container = self.shot
        if not isinstance(container, fdp.classes.shot.Shot):
            if isinstance(container._parent, fdp.classes.shot.Shot):
                print('Parsing {}'.format(container._name))
            else:
                print('Parsing {}.{}'.format(
                    container._parent._name, container._name))
        for attrname in dir(container):
            attr = getattr(container, attrname)
            if isContainer(attr):
                self.testSignalAxes(attr)
            elif isSignal(attr):
                self.assertTrue(hasattr(attr, 'axes'),
                                "Signal {} does not have 'axes' attr".format(attr._name))
                self.assertTrue(hasattr(attr, 'time'),
                                "Signal {} does not have 'time' attr".format(attr._name))
                self.assertTrue(isAxis(getattr(attr, 'time')),
                                "'time' attr is not axis object for signal {}".format(attr._name))
                for axisname in attr.axes:
                    self.assertTrue(hasattr(attr, axisname),
                                    "'axes' element {} not an attribute for signal {}".format(
                                    axisname, attr._name))
                    axis = getattr(attr, axisname)
                    self.assertTrue(isAxis(axis),
                                    "'axes' element {} is not axis object for signal {}".format(
                                    axisname, attr._name))
                for sigattrname in dir(attr):
                    sigattr = getattr(attr, sigattrname)
                    if isAxis(sigattr):
                        self.assertIn(sigattrname, attr.axes,
                                      "{} is axis but not in 'axes' attr for signal {}".format(
                                          sigattrname, attr._name))


if __name__ == '__main__':
    unittest.main()
