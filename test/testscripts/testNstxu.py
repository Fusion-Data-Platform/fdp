# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:12:13 2016

@author: drsmith
"""

import unittest
import MDSplus as mds
import pymssql
import fdp
from setup import SetupNstxu

print('running tests in {}'.format(__file__))

class TestNstxu(SetupNstxu):
        
    def testNstxuMachine(self):
        """
        Assert fdp has 'nstx' and 'nstxu' attributes.
        Assert fdp.nstxu is subclass of Machine.
        """
        self.assertTrue(hasattr(fdp, 'nstxu'))
        self.assertTrue(hasattr(fdp, 'nstx'))
        self.assertTrue(isinstance(self.nstxu, fdp.classes.machine.Machine))
                                   
    def testS0Attribute(self):
        """
        Assert nstxu has 's0' attribute
        """
        self.assertTrue(hasattr(self.nstxu, 's0'))
        
    def testDiagnosticContainers(self):
        """
        Assert nstxu.s0 has diagnostic attributes
        Assert that diagnostic attributes are subclasses of container.Container
        """
        containers = ['bes',
                      'usxr',
                      'mse',
                      'chers',
                      'mpts',
                      'magnetics',
                      'equilibria',
                      'filterscopes',
                      'nbi',
                      'neutrons']
        for containername in containers:
            self.assertTrue(hasattr(self.nstxu.s0, containername))
            container = getattr(self.nstxu.s0, containername)
            self.assertTrue(issubclass(type(container), fdp.classes.container.Container))
        
    def testLogbookConnection(self):
        """
        Assert Logbook object
        Assert logbook connection is not none
        """
        logbook = self.nstxu._logbook
        self.assertTrue(isinstance(logbook, fdp.classes.logbook.Logbook))
        logbook._make_logbook_connection()
        self.assertIsNotNone(logbook._logbook_connection)
        self.assertTrue(isinstance(logbook._logbook_connection, pymssql.Connection))
    
    def testMdsConnection(self):
        """
        Assert that self._connections is list of mds.Connection objects
        """
        self.assertTrue(isinstance(self.nstxu._connections[0], mds.Connection))
        self.assertIsNotNone(self.nstxu._connections[0].socket)
        self.assertIsNotNone(self.nstxu._connections[0].hostspec)
                                   
if __name__ == '__main__':
    unittest.main()
