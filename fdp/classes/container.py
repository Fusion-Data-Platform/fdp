# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:58:50 2015

@author: ktritz
"""
import os
import inspect
import types
import numpy as np
import xml.etree.ElementTree as ET

from .fdp_globals import FDP_DIR, VERBOSE
from . import parse
from .node import Node
from .signal import Signal

_tree_dict = {}


def init_class(cls, module_tree, **kwargs):
    cls._name = module_tree.get('name')
    if VERBOSE: print('init_class({})'.format(cls._name))
    if cls not in cls._instances:
        cls._instances[cls] = {}

    for read_only in ['root', 'container', 'classparent']:
        try:
            setattr(cls, '_'+read_only, kwargs[read_only])
            # print(cls._name, read_only, kwargs.get(read_only, 'Not there'))
        except:
            pass

    for item in ['mdstree', 'mdspath', 'units']:
        getitem = module_tree.get(item)
        if getitem is not None:
            setattr(cls, '_'+item, getitem)

    cls._base_items = set(cls.__dict__.keys())
    if VERBOSE: print('init_class({})  Calling parse_method({})'.
                      format(cls._name, cls._name))
    parse.parse_method(cls)


# TODO: odd circulat dependence: Container.__getattr__ call Factory() and
#       Factory() creates subclass of Container
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
        if VERBOSE: print('    {}.__init__()'.format(self._name))
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

        if VERBOSE: print('    {}.__init__() Begin module parsing'.format(self._name))

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
            method_defaults, defaults_dict = parse.parse_defaults(element)
            if hasattr(self._parent, method_defaults):
                defaults_dict.update(getattr(self._parent, method_defaults))
            setattr(self, method_defaults, defaults_dict)

        for element in module_tree.findall('axis'):
            if VERBOSE: print('    {}.__init__ Begin parsing axis {}'.
                              format(self._name,element.get('name')))
            signal_list = parse.parse_signal(self, element)
            branch_str = self._get_branchstr()
            for signal_dict in signal_list:
                SignalClassName = ''.join(['Axis', branch_str])
                if SignalClassName not in cls._classes:
                    SignalClass = type(SignalClassName, (Signal, cls), {})
                    parse.parse_method(SignalClass)
                    cls._classes[SignalClassName] = SignalClass
                else:
                    SignalClass = cls._classes[SignalClassName]
                SignalObj = SignalClass(**signal_dict)
                refs = parse.parse_refs(self, element, SignalObj._transpose)
                if not refs:
                    refs = SignalObj.axes
                for axis, ref in zip(SignalObj.axes, refs):
                    setattr(SignalObj, axis, getattr(self, '_'+ref))
                setattr(self, ''.join(['_', signal_dict['_name']]), SignalObj)
            if VERBOSE: print('    {}.__init__ End parsing axis {}'.
                              format(self._name,element.get('name')))

        for branch in module_tree.findall('container'):
            name = branch.get('name')
            branch_str = self._get_branchstr()
            ContainerClassName = ''.join(['Container', branch_str,
                                          name.capitalize()])
            if ContainerClassName not in cls._classes:
                ContainerClass = type(ContainerClassName, (cls, Container), {})
                init_class(ContainerClass, branch, classparent=cls)
                cls._classes[ContainerClassName] = ContainerClass
            else:
                ContainerClass = cls._classes[ContainerClassName]
            ContainerObj = ContainerClass(branch, parent=self)
            setattr(self, name, ContainerObj)
            self._containers[name] = ContainerObj

        for element in module_tree.findall('signal'):
            if VERBOSE: print('    {}.__init__ Begin parsing signal {}'.
                              format(self._name,element.get('name')))
            signal_list = parse.parse_signal(self, element)
            branch_str = self._get_branchstr()
            for signal_dict in signal_list:
                # name = element.get('name').format('').capitalize()
                SignalClassName = ''.join(['Signal', branch_str])
                if SignalClassName not in cls._classes:
                    SignalClass = type(SignalClassName, (Signal, cls), {})
                    parse.parse_method(SignalClass)
                    cls._classes[SignalClassName] = SignalClass
                else:
                    SignalClass = cls._classes[SignalClassName]
                SignalObj = SignalClass(**signal_dict)
                refs = parse.parse_refs(self, element, SignalObj._transpose)
                if not refs:
                    refs = SignalObj.axes
                for axis, ref in zip(SignalObj.axes, refs):
                    setattr(SignalObj, axis, getattr(self, '_'+ref))
                for default in element.findall('defaults'):
                    method_defaults, defaults_dict = parse.parse_defaults(default)
                    if hasattr(self, method_defaults):
                        defaults_dict.update(getattr(self, method_defaults))
                    setattr(SignalObj, method_defaults, defaults_dict)
                setattr(self, signal_dict['_name'], SignalObj)
                self._signals[signal_dict['_name']] = SignalObj
            if VERBOSE: print('    {}.__init__ End parsing signal {}'.
                              format(self._name,element.get('name')))

        if top and hasattr(self, '_preprocess'):
            self._preprocess()

        if VERBOSE: print('    {}.__init__() End module parsing'.format(self._name))

    def __getattr__(self, attribute):
        if VERBOSE: print('    {}.__getattr__({})'.format(self._name, attribute))
        try:
            if self._dynamic_containers[attribute] is None:
                branch_path = '.'.join([self._get_branch(), attribute])
                if VERBOSE:
                    print('    {}.__getattr__({}) calling Factory()'.
                          format(self._name, attribute))
                self._dynamic_containers[attribute] = \
                    Factory(branch_path, root=self._root,
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
        if 'Shot' in str(type(attr)):
            raise AttributeError("Attribute '{}' not found".format(attribute))
        if Container in attr.__class__.mro() and attribute[0] is not '_':
            raise AttributeError("Attribute '{}' not found".format(attribute))
        if inspect.ismethod(attr):
            if VERBOSE:
                print('    {}.__getattr__({}) returning parent method'.
                      format(self._name, attribute))
            return types.MethodType(attr.__func__, self)
        else:
            if VERBOSE:
                print('    {}.__getattr__({}) returning parent attr'.
                      format(self._name, attribute))
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


def Factory(module_branch, root=None, shot=None, parent=None):
    global _tree_dict

    """
    Factory method
    """

    if VERBOSE: print('Factory({}, root={}, shot={}, parent={})'.
                      format(module_branch, root, shot, parent))
    module_branch = module_branch.lower()
    module_list = module_branch.split('.')
    module = module_list[-1]
    branch_str = ''.join([word.capitalize() for word in module_list])
    if module_branch not in _tree_dict:
        module_path = os.path.join(FDP_DIR, 'modules', root._name,
                                   *module_list)
        parse_tree = ET.parse(os.path.join(module_path,
                                           ''.join([module, '.xml'])))
        module_tree = parse_tree.getroot()
        _tree_dict[module_branch] = module_tree
    try:
        ContainerClassName = 'Container' + branch_str
        if ContainerClassName not in Container._classes:
            ContainerClass = type(ContainerClassName, (Container,), {})
            if VERBOSE: print('Factory() calling init_class()')
            init_class(ContainerClass, _tree_dict[module_branch], root=root,
                       container=module, classparent=parent.__class__)
            Container._classes[ContainerClassName] = ContainerClass
        else:
            ContainerClass = Container._classes[ContainerClassName]

        return ContainerClass(_tree_dict[module_branch], shot=shot,
                              parent=parent, top=True)
    except:
        raise