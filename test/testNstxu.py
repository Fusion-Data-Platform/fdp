# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:12:13 2016

@author: drsmith
"""

import unittest
import fdp

print('running tests in {}'.format(__file__))


class TestSetup(unittest.TestCase):
    """
    Unit test set-up code with nstxu object and shot attribute
    """
    
    def setUp(self, shotnumber=204620):
        self.nstxu = fdp.nstx()
        self.shotnumber = shotnumber
        self.shot = getattr(self.nstxu, 's'+repr(self.shotnumber))
        
class TestNstxu(TestSetup):
        
    def testNstxuClass(self):
        """
        Ensure that fdp has 'nstx' and 'nstxu' attributes.
        Ensure that fdp.nstxu is subclass of Machine.
        """
        self.assertTrue(hasattr(fdp, 'nstxu'))
        self.assertTrue(hasattr(fdp, 'nstx'))
        self.assertTrue(issubclass(type(self.nstxu), 
                                   fdp.classes.machine.Machine))
                                   
    def testS0Attribute(self):
        """
        Test if nstxu has 's0' attribute
        """
        self.assertTrue(hasattr(self.nstxu, 's0'))
                                   
class TestNstxuShot(TestSetup):
    
    def testShotClass(self):
        self.assertTrue(hasattr(self.nstxu, 's'+repr(self.shotnumber)))
        self.assertTrue(issubclass(type(self.shot),
                                   fdp.classes.shot.Shot))
                                   
    def testDiagnsticContainers(self):
        """
        Ensure that all shot attributes are containers
        """
        diagnostics = dir(self.shot)
        for diagnostic in diagnostics:
            diag = getattr(self.shot, diagnostic)
            self.assertTrue(issubclass(type(diag),
                                       fdp.classes.container.Container))


if __name__ == '__main__':
    unittest.main()
