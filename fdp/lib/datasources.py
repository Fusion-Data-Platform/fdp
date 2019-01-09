#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:49:36 2017

@author: drsmith
"""

import os
from .globals import FdpError


def canonicalMachineName(machine=''):
    aliases = {'nstxu': ['nstx', 'nstxu', 'nstx-u'],
               'diiid': ['diiid', 'diii-d', 'd3d'],
               'cmod': ['cmod', 'c-mod']}
    for key, value in aliases.items():
        if machine.lower() in value:
            return key
    # invalid machine name
    raise FdpError('"{}" is not a valid machine name\n'.format(machine))


MDS_SERVERS = {
    'nstxu': {'hostname': 'skylark.pppl.gov',
              'port': '8000'},
    'diiid': {'hostname': 'atlas.gat.com',
              'port': '8000'}
}

EVENT_SERVERS = {
    'nstxu': {'hostname': 'skylark.pppl.gov',
              'port': '8000'},
    'diiid': {'hostname': 'atlas.gat.com',
              'port': '8000'},
    'ltx': {'hostname': 'lithos.pppl.gov',
            'port': '8000'}
}

LOGBOOK_CREDENTIALS = {
    'nstxu': {'server': 'sql2008.pppl.gov',
              'instance': None,
              'username': None,
              'password': None,
              'database': None,
              'port': '62917',
              'table': 'entries',
              'loginfile': os.path.join(os.getenv('HOME'),
                                        'nstxlogs.sybase_login')
              }
}
