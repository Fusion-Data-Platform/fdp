# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 11:10:48 2016

@author: drsmith
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import ttk

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
        
        self.figure = mpl.figure.Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        
        self.controlframe = ttk.Frame(master=self.tkroot, borderwidth=5, 
                                     width=100, height=100, relief='ridge')
        self.controlframe.pack(side='left')
        self.figureframe = ttk.Frame(master=self.tkroot, borderwidth=5, 
                                     width=100, height=100, relief='ridge')
        self.figureframe.pack(side='right')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figureframe)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        
        self.tkroot.mainloop()
        