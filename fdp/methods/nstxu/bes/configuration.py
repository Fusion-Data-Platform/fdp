# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:54:54 2016

@author: drsmith
"""
from __future__ import print_function

import os.path
import xml.etree.ElementTree as ET
from warnings import warn

from ....lib.globals import FdpError, FdpWarning
from ....lib.utilities import isContainer


def loadConfig(container=None):
    """
    """
    if not isContainer(container):
        raise FdpError(
            "loadConfig() is a BES container method, not signal method")
    config_file = os.path.join(os.path.dirname(__file__), 'configuration.xml')
    tree = ET.parse(config_file)
    root = tree.getroot()
    shot = container.shot
    configname = None
    for shotrange in root:
        if shotrange.tag == 'shotrange':
            start = int(shotrange.attrib['start'])
            stop = int(shotrange.attrib['stop'])
            if shot >= start and shot <= stop:
                configname = shotrange.attrib['config']
                break
    if configname is None:
        warn("Invalid shot for configuration", FdpWarning)
        return
    for config in root:
        if config.tag == 'config' and config.attrib['name'] == configname:
            for channel in config:
                signal = getattr(container, channel.attrib['name'])
                signal.row = int(channel.attrib['row'])
                signal.column = int(channel.attrib['column'])
            print('BES configuration loaded')
            return
    warn("BES configuration name not found", FdpWarning)
    return
