#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:49:36 2017

@author: drsmith
"""

import os
from .fdp_globals import FdpError


MACHINES = ['nstxu', 'diiid', 'cmod']

MDS_SERVERS = {
    'nstxu': {'hostname':'skylark.pppl.gov',
              'port':'8501'}
}

EVENT_SERVERS = {
    'nstxu': {'hostname':'skylark.pppl.gov',
              'port':'8501'},
    'ltx': {'hostname':'lithos.pppl.gov',
            'port':'8000'}
}

LOGBOOK_CREDENTIALS = {
    'nstxu': {'hostname': 'sql2008.pppl.gov',
              'server': 'sql2008.pppl.gov\sql2008',
              'username': os.getenv('USER') or os.getenv('USERNAME'),
              'password': 'pfcworld',
              'database': 'nstxlogs',
              'port': '62917',
              'table': 'entries'}
}


def machineAlias(machine):

    aliases = {'nstxu': ['nstx', 'nstxu', 'nstx-u'],
               'diiid': ['diiid', 'diii-d', 'd3d'],
               'cmod': ['cmod', 'c-mod']}

    for key in iter(aliases):
        if machine.lower() in aliases[key]:
            return key
    # invalid machine name
    txt = '"{}" is not a valid machine name\n'.format(machine)
    txt = txt + 'Valid machines are:\n'
    for values in aliases.itervalues():
        txt = txt + '  {}\n'.format(values)
    raise FdpError(txt)
