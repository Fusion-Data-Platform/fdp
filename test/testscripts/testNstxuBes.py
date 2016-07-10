# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 21:14:25 2016

@author: drsmith
"""

import unittest
import fdp
from setup import SetupNstxu

print('running tests in {}'.format(__file__))

class TestBes(SetupNstxu):
    
    def testContainerClass(self):
        self.assertTrue(hasattr(self.shot, 'bes'))
        self.assertTrue(isinstance(self.shot.bes,
                                   fdp.classes.container.Container))
                                   
    def testSignalClass(self):
        self.assertTrue(hasattr(self.shot.bes, 'ch01'))
        self.assertTrue(isinstance(self.shot.bes.ch01,
                                   fdp.classes.fdpsignal.Signal))

if __name__ == '__main__':
    unittest.main()
