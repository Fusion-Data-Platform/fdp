# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:35:36 2015

@author: ktritz
"""
import fdp_globals
from machine import Machine
import factory


class Fdp(object):
    """
    Top-level FDP object that contains machines.
    
    Exposed by re-mapping 'fdp' module in fdp.__init__.py.
    
    **Usage**::
    
        >>> import fdp
        >>> dir(fdp)
        ['cmod', 'diiid', 'nstx']
        >>> nstx = fdp.nstx
        
    """
    def __getattr__(self, attribute):
        machine_name = fdp_globals.machineAlias(attribute)
        if machine_name in fdp_globals.MACHINES:
            MachineClassName = ''.join(['Machine', machine_name.capitalize()])
            MachineClass = type(MachineClassName, (Machine, ), {})
            MachineClass._name = machine_name
            factory.parse_method(MachineClass, level='top')
            factory.parse_method(MachineClass, level=machine_name)
            machine = MachineClass(machine_name)
            setattr(self, machine_name, machine)
            return getattr(self, machine_name)
        raise

    def __dir__(self):
        return fdp_globals.MACHINES
