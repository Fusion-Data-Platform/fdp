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
        self.root = tk.Tk()
        self.root.title(title)
        self.mainframe = ttk.Frame(master=self.parent)
        
        self.controlframe = None
        self.figureframe = None
        
        self.root.mainloop()
        