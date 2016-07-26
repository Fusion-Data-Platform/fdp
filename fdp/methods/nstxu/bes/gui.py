# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:30:04 2016

@author: drsmith
"""

#from warnings import warn

from fdp.classes.gui import BaseGui
from fdp.classes.utilities import isSignal, isContainer
#from fdp.classes.fdp_globals import FdpWarning


class BesGui(BaseGui):
    
    def __init__(self, obj=None):
        super(BesGui, self).__init__(obj=obj, title='BES GUI')

    def plotObject(self):
        channels = ['ch01','ch09','ch17','ch25','ch33','ch41']
        for i, channel in enumerate(channels):
            ax = self.figure.add_subplot(2,3,i+1)
            ch = getattr(self.obj, channel)
            ch.plot(fig=self.figure, ax=ax)
        #self.axes = self.figure.add_subplot(2,3,1)
        #self.obj.ch01.plot(fig=self.figure, ax=self.axes)
        self.canvas.show()


def gui(obj):
    if isSignal(obj):
        return BaseGui(obj)
    if isContainer(obj):
        BesGui(obj)