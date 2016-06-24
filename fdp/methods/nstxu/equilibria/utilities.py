# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:42:07 2015

@author: ktritz
"""
import sys
import xml.etree.ElementTree as ET
import os
import importlib

module_dir = os.path.dirname(os.path.abspath(__file__))
FDP_DIR = os.path.join(module_dir, os.pardir, os.pardir, os.pardir)
class_dir = os.path.join(FDP_DIR, 'classes')
sys.path.insert(0, class_dir)
import container
import factory
sys.path.pop(0)

Container = container.Container
_tree_dict = factory._tree_dict
init_class = factory.init_class


def create_efit_objs(self):

    ContainerClassName = 'ContainerEquilibria'
    for efit in self.check_efit():
        branch = '.'.join(['equilibria', efit])
        # ContainerClassName = ''.join(['Equilibria', efit.capitalize()])
        if branch not in _tree_dict:
            filepath = os.path.join(FDP_DIR, 'modules', self._root._name,
                                    'equilibria', 'efit.xml')
            with open(filepath, 'r') as fileobj:
                xmlstring = fileobj.read()
            efitxml = xmlstring.replace('[efit]', efit)
            efitroot = ET.fromstring(efitxml)
            _tree_dict[branch] = efitroot

        if ContainerClassName not in Container._classes:
            ContainerClass = type(ContainerClassName, (Container,), {})
            init_class(ContainerClass, _tree_dict[branch], root=self._root,
                       container='equilibria', classparent=self.__class__)
            ContainerClass._name = 'efit'
            Container._classes[ContainerClassName] = ContainerClass
        else:
            ContainerClass = Container._classes[ContainerClassName]

        efitobj = ContainerClass(_tree_dict[branch], shot=self.shot,
                                 parent=self, mdstree=efit)

        efitobj._title = efit
        setattr(self, efit, efitobj)
        self._containers[efit] = efitobj
