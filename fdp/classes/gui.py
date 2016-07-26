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
import threading

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class BaseGui(threading.Thread):
    """
    Base GUI class for FDP GUIs.
    
    Class attributes enable setting shot/times across all GUIs derived from
    this base class.
    """

    basegui_shot = None
    basegui_tmin = None
    basegui_tmax = None
    basegui_update = False

    def __init__(self, signal=None, title='', parent=None, 
                 skipdefaultwidgets=False):
        super(BaseGui, self).__init__()
        self.title = title
        self.parent = parent
        self.signal = signal
        self.root = self.signal._root
        self.skipdefaultwidgets = skipdefaultwidgets
        self.start()

    def run(self):
        self.tkroot = tk.Tk()
        self.tkroot.title(self.title)
        
        
        self.controlframe = ttk.Frame(master=self.tkroot, borderwidth=3, 
                                      relief='ridge', padding=2)
        self.controlframe.pack(side='left', fill='y')
        # dummy frame to set controlframe width
        controlwidth = ttk.Frame(master=self.controlframe, width=125)
        controlwidth.pack(side='top', fill='x')
        
        if not self.skipdefaultwidgets:
            self.shotEntry = self.addEntry(text='Shot:  ')
            self.tminEntry = self.addEntry(text='Tmin (ms):  ')
            self.tmaxEntry = self.addEntry(text='Tmax (ms):  ')
            self.closeButton = self.addButton(text='Close', 
                                              command=self.tkroot.destroy)
        
        self.figure = mpl.figure.Figure()
        self.axes = self.figure.add_subplot(111)
        self.signal.plot(fig=self.figure, ax=self.axes)
        
        self.figureframe = ttk.Frame(master=self.tkroot, borderwidth=3,
                                     relief='ridge')
        self.figureframe.pack(side='left', expand=1, fill='both')
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figureframe)
        self.canvas.get_tk_widget().pack(expand=1, fill='both')
        self.canvas.show()
        
        self.tkroot.mainloop()
        
    def addEntry(self, text=None, width=8):
        frame = ttk.Frame(master=self.controlframe, padding=2)
        frame.pack(side='top', fill='x')
        label = ttk.Label(master=frame, text=text)
        label.pack(side='left')
        entry = ttk.Entry(master=frame, width=width)
        entry.pack(side='right')
        return entry
        
    def addButton(self, text=None, width=20, command=None):
        frame = ttk.Frame(master=self.controlframe, padding=2)
        frame.pack(side='top', fill='x')
        button = ttk.Button(master=frame, text=text, 
                            command=command,
                            width=width)
        button.pack(side='left')
        return button
