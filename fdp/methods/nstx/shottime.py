# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:34:23 2016

@author: ktritz
"""

def shottime(self):
    return float(self._netcat('skylark', 8530, ''))
