# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:42:07 2015

@author: ktritz
"""
from fdf.factory import _tree_dict, Container, init_class
import xml.etree.ElementTree as ET
from fdf.fdf_globals import FDF_DIR
import os


def _preprocess(self):

    ContainerClassName = 'ContainerEquilibria'
    for efit in self.check_efit():
        branch = '.'.join(['equilibria', efit])
        # ContainerClassName = ''.join(['Equilibria', efit.capitalize()])
        if branch not in _tree_dict:
            filepath = os.path.join(FDF_DIR, 'modules', 'equilibria',
                                    'efit.xml')
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
