# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:02:48 2016

@author: ktritz
"""

import socket

def _netcat(self, hostname, port, content):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    s.sendall(content)
    s.shutdown(socket.SHUT_WR)
    sdata = ""
    while 1:
        data = s.recv(1024)
        if data == "":
            break
        sdata = data
    s.close()
    return sdata
