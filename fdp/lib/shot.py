# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:14:03 2015

@author: ktritz
"""
from __future__ import print_function
from builtins import str, range
import inspect
import types
import numpy as np
from collections import MutableMapping

from .container import containerClassFactory


class Shot(MutableMapping):

#    _modules = None
    _logbook = None
    _machine = None

    def __init__(self, shot, machine):
        self.shot = shot

        # set class attributes if needed
        cls = self.__class__
        if cls._machine is None:
            cls._machine = machine
#        if cls._modules is None:
#            cls._modules = {module: None for module in self._machine._modules}
        if cls._logbook is None:
            cls._logbook = self._machine._logbook

        self._logbook_entries = self._logbook.get_entries(shot=self.shot)
        self._efits = []
        self._modules = {module: None for module in self._machine._modules}
        self.xp = self._get_xp()
        self.date = self._get_date()

    def _get_xp(self):
        # query logbook for XP, return XP (list if needed)
        xplist = []
        for entry in self._logbook_entries:
            if entry['xp']:
                xplist.append(entry['xp'])
        return list(set(xplist))

    def _get_date(self):
        # query logbook for rundate, return rundate
        if self._logbook_entries:
            return self._logbook_entries[0]['rundate']
        else:
            return

    def __getattr__(self, attr_name):
        if attr_name in self._modules:
            if self._modules[attr_name] is None:
                self._modules[attr_name] = containerClassFactory(attr_name,
                                                                 root=self._machine,
                                                                 shot=self.shot,
                                                                 parent=self)
            return self._modules[attr_name]
        else:
            try:
                attr = getattr(self._machine, attr_name)
            except AttributeError as e:
                # print('{} is not attribute of {}'.format(attr_name, self._machine._name))
                raise e
            if inspect.ismethod(attr):
                return types.MethodType(attr.__func__, self)
            else:
                return attr

    def __repr__(self):
        return '<Shot {}>'.format(self.shot)

    def __str__(self):
        return 'Shot {}'.format(self.shot)

    def __iter__(self):
        # return iter(self._modules.values())
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def __len__(self):
        return len(list(self._modules.keys()))

    def __delitem__(self, item):
        pass

    def __getitem__(self, item):
        return self._modules[item]

    def __setitem__(self, item, value):
        pass

    def __dir__(self):
        return list(self._modules.keys())

    def logbook(self):
        # show logbook entries
        if not self._logbook_entries:
            self._logbook_entries = self._logbook.get_entries(shot=self.shot)
        if self._logbook_entries:
            print('Logbook entries for {}'.format(self.shot))
            for entry in self._logbook_entries:
                print('************************************')
                print(('{shot} on {rundate} in XP {xp}\n'
                       '{username} in topic {topic}\n\n'
                       '{text}').format(**entry))
            print('************************************')
        else:
            print('No logbook entries for {}'.format(self.shot))

    def get_logbook(self):
        # return a list of logbook entries
        if not self._logbook_entries:
            self._logbook_entries = self._logbook.get_entries(shot=self.shot)
        return self._logbook_entries

    def check_efit(self):
        if len(self._efits):
            return self._efits
        trees = ['efit{}'.format(str(index).zfill(2)) for index in range(1, 7)]
        trees.extend(['lrdfit{}'.format(str(index).zfill(2))
                      for index in range(1, 13)])
        if self.shot == 0:
            return trees
        tree_exists = []
        for tree in trees:
            data = None
            connection = self._get_connection(self.shot, tree)
            try:
                data = connection.get('\{}::userid'.format(tree)).value
            except:
                pass
            if data and data is not '*':
                tree_exists.append(tree)
        self._efits = tree_exists
        return self._efits
