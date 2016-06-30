# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 21:14:25 2016

@author: drsmith
"""

import unittest
import fdp
from testNstxu import TestSetup

print('running tests in {}'.format(__file__))

class TestBes(TestSetup):
    
    def testContainerClass(self):
        self.assertTrue(hasattr(self.shot, 'bes'))
        self.assertTrue(issubclass(type(self.shot.bes),
                                   fdp.classes.container.Container))
                                   
    def testSignalClass(self):
        self.assertTrue(hasattr(self.shot.bes, 'd1ch01'))
        self.assertTrue(issubclass(type(self.shot.bes.d1ch01),
                                   fdp.classes.fdpsignal.Signal))

if __name__ == '__main__':
    unittest.main()
