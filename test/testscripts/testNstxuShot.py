# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 00:29:32 2016

@author: drsmith
"""

import unittest
import fdp
from .setup import SetupNstxu

print('running tests in {}'.format(__file__))

class TestNstxuShot(SetupNstxu):
    
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
                                       
    def testLogbook(self):
        """
        Assert logbook entries
        """
        pass
    
    def testTestShot(self):
        """
        Test shot with <6 digits
        """
        pass

if __name__ == '__main__':
    unittest.main()
