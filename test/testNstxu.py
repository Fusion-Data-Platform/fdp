# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:12:13 2016

@author: drsmith
"""

import unittest
import fdp

print('running tests in {}'.format(__file__))


class TestSetup(unittest.TestCase):
    
    def setUp(self):
        self.nstxu = fdp.nstx()
        self.shotnumber = 204620
        self.shot = getattr(self.nstxu, 's'+repr(self.shotnumber))
        

class TestNstxu(TestSetup):
        
    def testMachineClass(self):
        self.assertTrue(issubclass(type(self.nstxu), 
                                   fdp.classes.machine.Machine))
                                   
    def testShotClass(self):
        self.assertTrue(hasattr(self.shot, 's'+repr(self.shotnumber)))
        self.assertTrue(issubclass(type(self.shot),
                                   fdp.classes.shot.Shot))


if __name__ == '__main__':
    unittest.main()
