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

    def __init__(self, title='', parent=None, skipdefaultwidgets=False):
        self.parent = parent
        self.tkroot = tk.Tk()
        self.tkroot.title(title)
        
        self.controlframe = ttk.Frame(master=self.tkroot, borderwidth=3, 
                                      relief='ridge', padding=2)
        self.controlframe.pack(side='left', fill='y')
        # dummy frame to set controlframe width
        controlwidth = ttk.Frame(master=self.controlframe, width=125)
        controlwidth.pack(side='top', fill='x')
        
        if not skipdefaultwidgets:
            self.shotEntry = self.addEntry(text='Shot:  ')
            self.tminEntry = self.addEntry(text='Tmin (ms):  ')
            self.tmaxEntry = self.addEntry(text='Tmax (ms):  ')
        
        self.figureframe = ttk.Frame(master=self.tkroot, borderwidth=3,
                                     relief='ridge')
        self.figureframe.pack(side='left', expand=1, fill='both')
        
        self.figure = mpl.figure.Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figureframe)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(expand=1, fill='both')
        
        self.tkroot.mainloop()
        
    def addEntry(self, text=None, width=8):
        frame = ttk.Frame(master=self.controlframe, borderwidth=0,
                          relief='ridge', padding=2)
        frame.pack(side='top', fill='x')
        label = ttk.Label(master=frame, text=text)
        label.pack(side='left')
        entry = ttk.Entry(master=frame, width=width)
        entry.pack(side='right')
        return entry
        