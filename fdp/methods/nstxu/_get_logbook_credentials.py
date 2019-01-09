#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:19:06 2017

@author: drsmith
"""
from ...lib.datasources import LOGBOOK_CREDENTIALS

def _get_logbook_credentials(self):
    credentials = LOGBOOK_CREDENTIALS[self._name]
    with open(credentials['loginfile'], 'r') as f:
        f.readline() # 1st line is empty
        credentials['instance'] = f.readline().rstrip() # DB instance on 2nd line
        credentials['database'] = f.readline().rstrip() # database on 3rd line
        credentials['username'] = f.readline().rstrip() # username on 4th line
        credentials['password'] = f.readline().rstrip() # password on 5th line
    return credentials
