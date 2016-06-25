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
        

class TestNstxuModule(TestSetup):
        
    def testMachineClass(self):
        self.assertTrue(issubclass(type(self.nstxu), 
                                   fdp.classes.machine.Machine))
                                   
    def testShotClass(self):
        self.assertTrue(hasattr(self.shot, 's'+repr(self.shotnumber)))
        self.assertTrue(issubclass(type(self.shot),
                                   fdp.classes.shot.Shot))


class TestBesModule(TestSetup):
    
    def testContainerClass(self):
        self.assertTrue(hasattr(self.shot, 'bes'))
        self.assertTrue(issubclass(type(self.shot.bes),
                                   fdp.classes.container.Container))
                                   
    def testSignalClass(self):
        print(dir(self.shot.bes))
        self.assertTrue(hasattr(self.shot.bes, 'd1ch01'))
        self.assertTrue(issubclass(type(self.shot.bes.d1ch01),
                                   fdp.classes.fdpsignal.Signal))

if __name__ == '__main__':
    unittest.main()
