# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:35:36 2015

@author: ktritz
"""
from . import fdp_globals
from .machine import Machine
from .factory import parse_method


class Fdp(object):
    """
    Top-level container for machines.
    
    An Fdp instance is mapped to 'fdp' module in fdp.__init__.py.
    
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
            parse_method(MachineClass, level='top')
            parse_method(MachineClass, level=machine_name)
            machine = MachineClass(machine_name)
            setattr(self, machine_name, machine)
            return getattr(self, machine_name)
        raise

    def __dir__(self):
        return fdp_globals.MACHINES
