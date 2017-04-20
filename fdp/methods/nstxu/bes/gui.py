# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:30:04 2016

@author: drsmith
"""

from ....classes.gui import BaseGui
from ....classes.utilities import isSignal, isContainer


class BesGui(BaseGui):

    def __init__(self, obj=None):
        super(BesGui, self).__init__(obj=obj, title='BES GUI')

    def plotObject(self):
        channels = ['ch01', 'ch09', 'ch17',
                    'ch25', 'ch33', 'ch41']
        for i, channel in enumerate(channels):
            ax = self.figure.add_subplot(2, 3, i + 1)
            ch = getattr(self.obj, channel)
            ch.plot(fig=self.figure, ax=ax)
        self.canvas.show()


def gui(obj):
    if isSignal(obj):
        return BaseGui(obj)
    if isContainer(obj):
        return BesGui(obj)
