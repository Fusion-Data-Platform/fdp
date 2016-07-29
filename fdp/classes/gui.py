# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 11:10:48 2016

@author: drsmith
"""

import sys
from warnings import warn
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
import ttk
import threading
import time

import matplotlib as mpl
#mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .fdp_globals import FdpWarning


class BaseGui(threading.Thread):
    """
    Base GUI class for FDP GUIs.
    
    Class attributes enable setting shot/times across all GUIs derived from
    this base class.
    """

    def __init__(self, obj=None, title='', parent=None, 
                 defaultwidgets=True):
        super(BaseGui, self).__init__()
        self.title = title
        self.parent = parent
        self.obj = obj
        self.root = self.obj._root
        self.shotkeys = sorted(self.root._shots.keys())
        self.defaultwidgets = defaultwidgets
        self.start()

    def run(self):
        self.tkroot = tk.Tk()
        self.tkroot.title(self.title)
        self.controlframe = ttk.Frame(master=self.tkroot, 
                                      borderwidth=3, 
                                      relief='ridge', 
                                      padding=2)
        self.controlframe.pack(side='left', fill='y')
        # dummy frame to set controlframe width
        controlwidth = ttk.Frame(master=self.controlframe, width=125)
        controlwidth.pack(side='top', fill='x')
        
        if self.defaultwidgets:
            self.addDefaultWidgets()
        
        self.figure = mpl.figure.Figure()
        self.figureframe = ttk.Frame(master=self.tkroot, 
                                     borderwidth=3,
                                     relief='ridge')
        self.figureframe.pack(side='left', expand=1, fill='both')
        self.canvas = FigureCanvasTkAgg(self.figure, 
                                        master=self.figureframe)
        self.canvas.get_tk_widget().pack(expand=1, fill='both')
        
        self.plotObject()
        
        self.tkroot.after(5000, self.checkShotKeys)
        self.tkroot.mainloop()
        
    def checkShotKeys(self):
        newkeys = sorted(self.root._shots.keys())
        if self.shotkeys != newkeys:
            for i, key in enumerate(newkeys):
                if key not in self.shotkeys:
                    self.shotList.insert(i, key)
            self.shotkeys = newkeys
        self.tkroot.after(1000, self.checkShotKeys)
        
    def addDefaultWidgets(self):
        self.shotEntry = self.insertButtonEntry(text='Add shot',
                                                command=self.addShot)
        self.shotEntry.bind('<Return>', self.shotEntryEvent)
        self.insertShotListbox()
        self.tminEntry = self.insertTextEntry(text='Tmin (ms):  ')
        self.tmaxEntry = self.insertTextEntry(text='Tmax (ms):  ')
        self.closeButton = self.insertButton(text='Close', 
                                             command=self.tkroot.destroy)
        self.printButton = self.insertButton(text='Save', 
                                             command=None)
        
    def shotEntryEvent(self, event):
        self.addShot()
        
    def addShot(self):
        try:
            shot = int(self.shotEntry.get())
        except ValueError:
            warn('Shot value is invalid', FdpWarning)
            return
        self.root.addshot(shot)
        self.checkShotKeys()
    
    def plotObject(self):
        self.axes = self.figure.add_subplot(111)
        self.obj.plot(fig=self.figure, ax=self.axes)
        self.canvas.show()
        
    def insertButtonEntry(self, text=None, width=8, command=None):
        frame = ttk.Frame(master=self.controlframe, padding=2)
        frame.pack(side='top', fill='x')
        button = ttk.Button(master=frame, 
                            text=text,
                            command=command)
        button.pack(side='left')
        entry = ttk.Entry(master=frame, width=width)
        entry.pack(side='right')
        return entry
        
    def insertTextEntry(self, text=None, width=8):
        frame = ttk.Frame(master=self.controlframe, padding=2)
        frame.pack(side='top', fill='x')
        label = ttk.Label(master=frame, text=text)
        label.pack(side='left')
        entry = ttk.Entry(master=frame, width=width)
        entry.pack(side='right')
        return entry

    def insertShotListbox(self):
        frame = ttk.Frame(master=self.controlframe, padding=2)
        frame.pack(side='top', fill='x')
        label = ttk.Label(master=frame, text='Shots:')
        label.pack(side='left', anchor=tk.N)
        self.shotControlVar = tk.StringVar()
        self.shotList = tk.Listbox(master=frame,
                                   listvariable=self.shotControlVar,
                                   selectmode=tk.SINGLE,
                                   height=5, width=8)
        for key in self.shotkeys:
            self.shotList.insert(tk.END, key)
        self.shotList.pack(side='right', anchor=tk.N)
        
    def insertButton(self, text=None, width=20, command=None):
        frame = ttk.Frame(master=self.controlframe, padding=2)
        frame.pack(side='bottom', fill='x')
        button = ttk.Button(master=frame, text=text, 
                            command=command,
                            width=width)
        button.pack(side='left')
        return button
