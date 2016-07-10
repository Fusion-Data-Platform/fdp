# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:12:13 2016

@author: drsmith
"""

import unittest
import fdp
from .setup import SetupNstxu

print('running tests in {}'.format(__file__))

class TestNstxu(SetupNstxu):
        
    def testNstxuMachine(self):
        """
        Assert fdp has 'nstx' and 'nstxu' attributes.
        Assert fdp.nstxu is subclass of Machine.
        """
        self.assertTrue(hasattr(fdp, 'nstxu'))
        self.assertTrue(hasattr(fdp, 'nstx'))
        self.assertTrue(issubclass(type(self.nstxu), 
                                   fdp.classes.machine.Machine))
                                   
    def testS0Attribute(self):
        """
        Assert nstxu has 's0' attribute
        """
        self.assertTrue(hasattr(self.nstxu, 's0'))
        
    def testLogbookConnection(self):
        """
        Assert Logbook object
        Assert logbook connection is not none
        """
        pass
    
    def testMdsConnection(self):
        """
        Assert that self._connections is list of mds.Connection objects
        """
        pass
                                   
if __name__ == '__main__':
    unittest.main()
