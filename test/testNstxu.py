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
        self.assertTrue(hasattr(fdp, 'nstxu'), 'fdp.nstxu() missing')
        self.assertTrue(hasattr(fdp, 'nstx'), 'fdp.nstx() missing')
        self.assertTrue(isinstance(self.nstxu, fdp.classes.machine.Machine),
                        'fdp.nstxu() does not return machine.Machine instance')
                                   
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
            self.assertTrue(hasattr(self.nstxu.s0, containername),
                            'nstxu.s0 missing {} attribute'.format(
                            containername))
            container = getattr(self.nstxu.s0, containername)
            self.assertTrue(issubclass(type(container),
                                       fdp.classes.container.Container),
                            '{} is not container.Container subclass'.format(
                            repr(container)))
        
    def testLogbookConnection(self):
        """
        Assert Logbook object
        Assert logbook connection is not none
        """
        logbook = self.nstxu._logbook
        self.assertTrue(isinstance(logbook, fdp.classes.logbook.Logbook),
                        'nstxu._logbook is not logbook.Logbook instance')
        logbook._make_logbook_connection()
        self.assertIsNotNone(logbook._logbook_connection,
                             'nstxu._logbook._logbook_connection is None')
        self.assertTrue(isinstance(logbook._logbook_connection, 
                                   pymssql.Connection),
                        'nstxu._logbook._logbook_connection \
                        is not pymssql.Connection instance')
    
    def testMdsConnection(self):
        """
        Assert that self._connections is list of mds.Connection objects
        """
        self.assertTrue(isinstance(self.nstxu._connections[0], mds.Connection),
                        'nstxu._connections[0] is not mds.Connection instance')
        self.assertIsNotNone(self.nstxu._connections[0].socket,
                             'nstxu._connections[0].socket is None')
        self.assertIsNotNone(self.nstxu._connections[0].hostspec,
                             'nstxu._connections[0].hostspec is None')
                                   
if __name__ == '__main__':
    unittest.main()
