# -*- coding: utf-8 -*-
"""
Package-level attributes, methods, and FdfError class

Created on Thu Jun 18 11:18:16 2015

@author: ktritz
"""
import os

MACHINES = ['nstxu', 'diiid', 'cmod']

FDP_DIR = os.path.dirname(os.path.abspath(__file__))
FDP_DIR = os.path.join(FDP_DIR, os.path.pardir)
"""Path string: top-level directory for FDF package"""

MDS_SERVERS = {
    'nstxu': 'skylark.pppl.gov:8501'
}

EVENT_SERVERS = {
    'nstxu': 'skylark.pppl.gov:8501',
    'ltx': 'lithos.pppl.gov:8000'
}
"""Dictionary: machine-name key paired to MDS server"""

LOGBOOK_CREDENTIALS = {
    'nstxu': {
        'server': 'sql2008.pppl.gov\sql2008',
        'username': os.getenv('USER') or os.getenv('USERNAME'),
        'password': 'pfcworld',
        'database': 'nstxlogs',
        'port': '62917',
        'table': 'entries'
    }
}
"""Dictionary: machine-name key paired with logbook login credentials"""


def machineAlias(machine):

    aliases = {
        'nstxu': ['nstx', 'nstxu', 'nstx-u'],
        'diiid': ['diiid', 'diii-d', 'd3d'],
        'cmod': ['cmod', 'c-mod']
    }

    for key, value in iter(aliases.items()):
        if machine.lower() in value:
            return key
    txt = '{} is not a valid machine; valid machines are:\n'.format(machine)
    for machinekey in aliases:
        txt = txt + '  {}\n'.format(machinekey)
    raise FdpError(txt)


class FdpError(Exception):
    """
    Error class for FDF package

    **Usage**::

        raise FdfError('my error message')

    """
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message
