# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:42:54 2016

@author: ktritz
"""

def currentshot(self):
    return self._connections[0].get('current_shot("nstx")').value
