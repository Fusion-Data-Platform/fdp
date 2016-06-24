# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:14:03 2015

@author: ktritz
"""
import inspect
import types
import numpy as np
from collections import MutableMapping
from . import factory
from .container import Container


class Shot(MutableMapping):

    def __init__(self, shot, root=None, parent=None):
        self.shot = shot
        self._shotobj = self
        self._root = root
        self._parent = parent
        self._logbook = root._logbook
        self._logbook_entries = []
        self._modules = {module: None for module in root._get_modules()}
        self.xp = self._get_xp()
        self.date = self._get_date()
        self._efits = []

    def __getattr__(self, attribute):
        # first see if the attribute is in the Machine object
        try:
            attr = getattr(self._parent, attribute)
            if inspect.ismethod(attr):
                return types.MethodType(attr.im_func, self)
            else:
                return attr
        except:
            pass  # failed, so check other locations
        if attribute in self._modules:
            if self._modules[attribute] is None:
                self._modules[attribute] = factory.Factory(attribute, Container, root=self._root,
                                                   shot=self.shot, parent=self)
            return self._modules[attribute]
        raise AttributeError("{} shot: {} has no attribute '{}'".format(
                                 self._root._name, self.shot, attribute))

    def __repr__(self):
        return '<Shot {}>'.format(self.shot)

    def __iter__(self):
        # return iter(self._modules.values())
        return iter(self._modules)

    def __contains__(self, value):
        return value in self._modules

    def __len__(self):
        return len(self._modules.keys())

    def __delitem__(self, item):
        pass

    def __getitem__(self, item):
        return self._modules[item]

    def __setitem__(self, item, value):
        pass

    def __dir__(self):
        return self._modules.keys()

    def _get_xp(self):
        # query logbook for XP, return XP (list if needed)
        if not self._logbook_entries:
            self._logbook_entries = self._logbook.get_entries(shot=self.shot)
        xplist = []
        for entry in self._logbook_entries:
            if entry['xp']:
                xplist.append(entry['xp'])
        if len(np.unique(xplist)) == 1:
            xp = xplist.pop(0)
        else:
            xp = np.unique(xplist)
        return xp

    def _get_date(self):
        # query logbook for rundate, return rundate
        if not self._logbook_entries:
            self._logbook_entries = self._logbook.get_entries(shot=self.shot)
        date = 0
        if self._logbook_entries:
            date = self._logbook_entries[0]['rundate']
        return date

    def logbook(self):
        # return a list of logbook entries (dictionaries)
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