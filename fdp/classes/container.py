# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:58:50 2015

@author: ktritz
"""
import os
import inspect
import types
import numpy as np
from . import fdp_globals
from . import factory
from .node import Node
from .fdpsignal import Signal


FDP_DIR = fdp_globals.FDP_DIR

class Container(object):
    """
    Container class
    """
    _instances = {}
    _classes = {}

    def __init__(self, module_tree, top=False, **kwargs):

        cls = self.__class__

        self._signals = {}
        self._axes = {}
        self._containers = {}
        self._dynamic_containers = {}
        self._tags = []
        self._title = module_tree.get('title')
        self._desc = module_tree.get('desc')

        for read_only in ['parent']:
            setattr(self, '_'+read_only, kwargs.get(read_only, None))

        try:
            self.shot = kwargs['shot']
            self._mdstree = kwargs['mdstree']
        except:
            pass

        if self.shot is not None:
            try:
                cls._instances[cls][self.shot].append(self)
            except:
                cls._instances[cls][self.shot] = [self]

        if top:
            self._get_dynamic_containers()

        for node in module_tree.findall('node'):
            branch_str = self._get_branchstr()
            NodeClassName = ''.join(['Node', branch_str])
            if NodeClassName not in cls._classes:
                NodeClass = type(NodeClassName, (Node, cls), {})
                cls._classes[NodeClassName] = NodeClass
            else:
                NodeClass = cls._classes[NodeClassName]
            # NodeClass._mdstree = parse_mdstree(self, node)
            setattr(self, node.get('name'), NodeClass(node, parent=self))

        for element in module_tree.findall('defaults'):
            method_defaults, defaults_dict = factory.parse_defaults(element)
            if hasattr(self._parent, method_defaults):
                defaults_dict.update(getattr(self._parent, method_defaults))
            setattr(self, method_defaults, defaults_dict)

        for element in module_tree.findall('axis'):
            signal_list = factory.parse_signal(self, element)
            branch_str = self._get_branchstr()
            for signal_dict in signal_list:
                SignalClassName = ''.join(['Axis', branch_str])
                if SignalClassName not in cls._classes:
                    SignalClass = type(SignalClassName, (Signal, cls), {})
                    factory.parse_method(SignalClass)
                    cls._classes[SignalClassName] = SignalClass
                else:
                    SignalClass = cls._classes[SignalClassName]
                SignalObj = SignalClass(**signal_dict)
                refs = factory.parse_refs(self, element, SignalObj._transpose)
                if not refs:
                    refs = SignalObj.axes
                for axis, ref in zip(SignalObj.axes, refs):
                    setattr(SignalObj, axis, getattr(self, '_'+ref))
                setattr(self, ''.join(['_', signal_dict['_name']]), SignalObj)

        for branch in module_tree.findall('container'):
            name = branch.get('name')
            branch_str = self._get_branchstr()
            ContainerClassName = ''.join(['Container', branch_str,
                                          name.capitalize()])
            if ContainerClassName not in cls._classes:
                ContainerClass = type(ContainerClassName, (cls, Container), {})
                factory.init_class(ContainerClass, branch, classparent=cls)
                cls._classes[ContainerClassName] = ContainerClass
            else:
                ContainerClass = cls._classes[ContainerClassName]
            ContainerObj = ContainerClass(branch, parent=self)
            setattr(self, name, ContainerObj)
            self._containers[name] = ContainerObj

        for element in module_tree.findall('signal'):
            signal_list = factory.parse_signal(self, element)
            branch_str = self._get_branchstr()
            for signal_dict in signal_list:
                # name = element.get('name').format('').capitalize()
                SignalClassName = ''.join(['Signal', branch_str])
                if SignalClassName not in cls._classes:
                    SignalClass = type(SignalClassName, (Signal, cls), {})
                    factory.parse_method(SignalClass)
                    cls._classes[SignalClassName] = SignalClass
                else:
                    SignalClass = cls._classes[SignalClassName]
                SignalObj = SignalClass(**signal_dict)
                refs = factory.parse_refs(self, element, SignalObj._transpose)
                if not refs:
                    refs = SignalObj.axes
                for axis, ref in zip(SignalObj.axes, refs):
                    setattr(SignalObj, axis, getattr(self, '_'+ref))
                for default in element.findall('defaults'):
                    method_defaults, defaults_dict = factory.parse_defaults(default)
                    if hasattr(self, method_defaults):
                        defaults_dict.update(getattr(self, method_defaults))
                    setattr(SignalObj, method_defaults, defaults_dict)
                setattr(self, signal_dict['_name'], SignalObj)
                self._signals[signal_dict['_name']] = SignalObj

        if top and hasattr(self, '_preprocess'):
            self._preprocess()

    def __getattr__(self, attribute):

        try:
            if self._dynamic_containers[attribute] is None:
                branch_path = '.'.join([self._get_branch(), attribute])
                self._dynamic_containers[attribute] = \
                    factory.Factory(branch_path, root=self._root,
                                    shot=self.shot, parent=self)

            return self._dynamic_containers[attribute]
        except KeyError:
            pass

        if not hasattr(self, '_parent') or self._parent is None:
            raise AttributeError("Attribute '{}' not found".format(attribute))

        if hasattr(self._parent, '_signals') and \
                attribute in self._parent._signals:
            raise AttributeError("Attribute '{}' not found".format(attribute))

        attr = getattr(self._parent, attribute)
        if Container in attr.__class__.mro() and attribute[0] is not '_':
            raise AttributeError("Attribute '{}' not found".format(attribute))
        if inspect.ismethod(attr):
            return types.MethodType(attr.im_func, self)
        else:
            return attr

    def _get_dynamic_containers(self):
        if len(self._dynamic_containers) is 0:
            container_dir = self._get_path()
            if not os.path.isdir(container_dir):
                return
            files = os.listdir(container_dir)
            self._dynamic_containers = {container: None for container in
                                   files if os.path.isdir(
                                   os.path.join(container_dir, container)) and
                                   container[0] is not '_'}

    @classmethod
    def _get_path(cls):
        branch = cls._get_branch().split('.')
        path = os.path.join(FDP_DIR, 'modules', cls._root._name)
        for step in branch:
            newpath = os.path.join(path, step)
            if not os.path.isdir(newpath):
                break
            path = newpath
        return path

    def __dir__(self):
        items = self.__dict__.keys()
        items.extend(self.__class__.__dict__.keys())
        if Signal not in self.__class__.mro():
            items.extend(self._dynamic_containers.keys())
        return [item for item in set(items).difference(self._base_items)
                if item[0] is not '_']

    def __iter__(self):
        if not len(self._signals):
            items = sorted(self._containers.values(), key= lambda obj: obj._name.lower())
            # items.extend(self._dynamic_containers.values())
        else:
            items = sorted(self._signals.values(), key= lambda obj: obj._name.lower())
        return iter(items)

    @classmethod
    def _get_branch(cls):
        if 'Shot' in str(cls):
            return None
        branch = cls._name
        parent = cls._classparent
        while 'Shot' not in str(parent) and 'Shot' not in str(parent.__class__):
            branch = '.'.join([parent._name, branch])
            parent = parent._classparent
        return branch

    @classmethod
    def _get_branchstr(cls):
        branch = cls._get_branch()
        return ''.join([sub.capitalize() for sub in branch.split('.')])

    @classmethod
    def _is_container(cls):
        return 'Container' in str(cls)

    @classmethod
    def _is_signal(cls):
        return 'Signal' in str(cls)

    @classmethod
    def _is_axis(cls):
        return 'Axis' in str(cls)

    @classmethod
    def _is_type(cls, obj_type):
        method_name = '_is_{}'.format(obj_type.lower())
        try:
            return getattr(cls, method_name)()
        except:
            return False

    def _contains(self, string):
        word_list = [s for s in [self._name, self._title] if s]
        word_list.extend(self._tags)
        return np.any([string.lower() in word.lower() for word in word_list])
