#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 12:10:25 2017

@author: drsmith
"""

import socket


def try_connection(host=None, port=None):
    try:
        s = socket.create_connection((host, port), 3)
        print('success: {} on port {}'.format(host, port))
        s.close()
        return True
    except Exception as ex:
        print('Exception for host {} on port {}: {}'.format(host, port, ex))
        return False

if __name__ == '__main__':
    connections = [('skylark.pppl.gov', 8501),
                   ('sql2008.pppl.gov', 62917)]
    for pair in connections:
        try_connection(pair[0], pair[1])