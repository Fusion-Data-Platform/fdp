# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 11:10:48 2016

@author: drsmith
"""

import sys
import ttk
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
plt.ioff()


class BaseGui(object):
    """
    Base GUI class for FDP GUIs.
    
    Class attributes enable setting shot/times across all GUIs derived from
    this base class.
    """

    global_shot = None
    global_tmin = None
    global_tmax = None
    global_update = False

    def __init__(self, title='', parent=None):
        self.parent = parent
        self.tkroot = tk.Tk()
        self.tkroot.title(title)
        
        self.mainframe = ttk.Frame(master=self.parent)
        
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tkroot)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.controlframe = None
        self.figureframe = None
        
        self.tkroot.mainloop()
        