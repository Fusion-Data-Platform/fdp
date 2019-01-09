# -*- coding: utf-8 -*-
"""
signals.py - module containing Signal class

**Classes**

* Signal - signal class for data objects

Created on Tue Jun 23 2015

@author: hyuh
"""
from __future__ import print_function
import sys
if sys.version_info > (3,):
    long = int

import inspect
import types
import numpy as np

from .globals import FdpError


class Signal(np.ndarray):
    """
    sig=fdp.Signal(signal_ndarray, units='m/s', axes=['radius','time'],
                   axes_values=[ax1_1Darray, ax2_1Darray],
                   axes_units=['s','cm'])

    e.g.:
    mds.Signal(np.arange((20*10)).reshape((10,20)), units='keV',
               axes=['radius','time'], axes_values=[100+np.arange(10)*5,
               np.arange(20)*0.1], axes_units=['s','cm'])

    or an empty signal:
    s=mds.Signal()
    default axes order=[time, space]
    sig=fdp.Signal(units='m/s', axes=['radius','time'],
                   axes_values=[radiusSignal, timeSignal])
    """
    def __new__(cls, input_array=[], **kwargs):
        obj = np.asanyarray(input_array).view(cls).copy()
        for key in iter(kwargs):
            setattr(obj, key, kwargs[key])
        return obj

    def __init__(self, **kwargs):
        # self.mdsshape = self._get_mdsshape()
        pass

    def __array_finalize__(self, obj):
        """
        see https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        self is the new object; obj is the original object
        type(self) is always Signal subclass
        type(obj) is None for explicit constructor like a = Signal(...)
                  is ndarray for "view casting"
                  is type(self) for slicing or copy
        """
        objdict = getattr(obj, '__dict__', None)
        if obj is None or objdict is None:
            return
        # logic for view casting and slicing/copy
        objaxes = getattr(obj, 'axes', None)
        objslic = getattr(obj, '_slic', None)
        for key in iter(objdict):
            if objaxes and key in objaxes:
                # skip copy of axis attributes
                pass
            elif key in ['axes', 'point_axes']:
                # shallow copy obj.axes and obj.point_axes
                setattr(self, key, objdict[key][:])
            else:
                setattr(self, key, objdict[key])
        if objdict.get('_fname') == 'transpose':
            if objaxes is not None:
                if '_fargs' in objdict:
                    self.axes = [obj.axes[i] for i in objdict['_fargs'][0]]
                else:
                    self.axes = obj.axes[::-1]
        # _deltmpattr = True
        # if objdict.get('_debug'):
        #     _deltmpattr = False
        if objaxes:
            for axis in objaxes:
                if objslic is not None:
                    # slice axis according to _slic
                    obj_axis = getattr(obj, axis)
                    if isinstance(objslic, (slice,list,np.ndarray)):
                        # logic for 1D arrays
                        setattr(self, axis, obj_axis[objslic])
                    elif isinstance(objslic, tuple):
                        # logic for multi-dim arrays
                        slic_axis = tuple([objslic[objaxes.index(axisaxis)] for
                                           axisaxis in (obj_axis.axes + [axis])])
                        if isinstance(slic_axis[0], (int, int, float, np.generic)):
                            # "point_axes" is a dict with axis keys and dict values
                            if axis in self.point_axes:
                                raise FdpError('Point axis already present')
                            self.point_axes.append({'axis': axis,
                                                    'value': obj_axis[slic_axis],
                                                    'units': obj_axis.units})
                            self.axes.remove(axis)
                        elif isinstance(slic_axis[0], slice):
                            setattr(self, axis, obj_axis[slic_axis])
                        else:
                            raise FdpError('slic_axis is unexpected type')
#                        for axisaxis in obj_axis.axes:
#                            if isinstance(objslic[objaxes.index(axisaxis)], (int, long, float, np.generic)):
#                                obj_axis.axes.remove(axisaxis)
                    else:
                        raise FdpError('obj._slic is unexpected type')
                else:
                    # obj._slic is undefined; copy each axis as is
                    setattr(self, axis, getattr(obj, axis, None))

        # clean-up temp attributes
        # def delattrtry(ob, at):
        #     try:
        #         delattr(ob, at)
        #     except:
        #         pass

        # if _deltmpattr:
        for attrname in ['_slic','_fname','_fargs','_fkwargs']:
            for o in [self, obj]:
                if hasattr(o, attrname):
                    delattr(o, attrname)
        # delattrtry(self, '_slic')
        # delattrtry(self, '_fname')
        # delattrtry(self, '_fargs')
        # delattrtry(self, '_fkwargs')
        # delattrtry(obj, '_slic')
        # delattrtry(obj, '_fname')
        # delattrtry(obj, '_fargs')
        # delattrtry(obj, '_fkwargs')

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_prepare__(self, out_arr, context=None):
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __getitem__(self, index):
        '''
        self must be Signal class for this to be called, so therefore
        must have the _slic attribute. The _slic attribute preserves indexing for attributes
        '''
        # This passes index to array_finalize after a new signal obj is created
        # to assign axes
        def parseindex(index, dims):
            # format index to account for single elements and pad with appropriate slices.
            #int2slc=lambda i: slice(-1,-2,-1) if int(i) == -1 else slice(int(i),int(i)+1)
            if isinstance(index, (list, slice, np.ndarray)):
                # index is list, slice, or ndarray
                if dims < 2:
                    return index
                else:
                    newindex = [index]
            elif isinstance(index, (int, int, float, np.generic)):
                newindex = [int(index)]
            elif isinstance(index, tuple):
                newindex = [int(i) if isinstance(i, (int, int, float, np.generic))
                            else i for i in index]
            # check for ellipses in newindex
            ellipsisbool = [Ellipsis is i for i in newindex]
            if sum(ellipsisbool) > 0:
                # elipses exists
                ellipsisindex = ellipsisbool.index(True)
                slcpadding = ([slice(None)] * (dims - len(newindex) + 1))
                newindex = newindex[:ellipsisindex] \
                    + slcpadding \
                    + newindex[ellipsisindex + 1:]
            else:
                # no elipses
                newindex = newindex + ([slice(None)] * (dims - len(newindex)))
            return tuple(newindex)

        slcindex = parseindex(index, self.ndim)
        self._slic = slcindex
        if self._empty is True:
            self._get_mdsdata()
        return super(Signal, self).__getitem__(slcindex)

    def _get_mdsdata(self):
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self._empty = False
            self[:] = data

    def _get_mdsshape(self):
        return self._root._get_mdsshape(self)

    def __getattr__(self, attribute):
        if attribute is '_parent' or self._parent is None:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        attr = getattr(self._parent, attribute)
        if inspect.ismethod(attr):
            return types.MethodType(attr.__func__, self)
        else:
            return attr

    def __repr__(self):
#        self._get_mdsdata()
        return super(Signal, self).__repr__()

    def __str__(self):
#        self._get_mdsdata()
        return super(Signal, self).__str__()

    def __getslice__(self, start, stop):
        """
        This solves a subtle bug, where __getitem__ is not called, and all
        the dimensional checking not done, when a slice of only the first
        dimension is taken, e.g. a[1:3]. From the Python docs:
        Deprecated since version 2.0: Support slice objects as parameters
        to the __getitem__() method. (However, built-in types in CPython
        currently still implement __getslice__(). Therefore, you have to
        override it in derived classes when implementing slicing.)
        """
        return self.__getitem__(slice(start, stop))

    def __call__(self, **kwargs):
        slc = [slice(None)] * len(self.axes)
        for axis_name, axis_values in kwargs.items():
            if axis_name not in self.axes:
                print('      {} is not a valid axis.'.format(axis_name))
                raise TypeError
            iaxis = self.axes.index(axis_name)
            axis = getattr(self, axis_name)
            try:
                axis_indices = [np.abs(value - axis[:]).argmin()
                             for value in axis_values]
                slc[iaxis] = slice(axis_indices[0], axis_indices[1])
            except TypeError:
                axis_indices = np.abs(axis_values - axis[:]).argmin()
                slc[iaxis] = axis_indices
        return self[tuple(slc)]

    def __bool__(self):
        return bool(self.mdsshape)

    def sigwrapper(f):
        def inner(*args, **kwargs):
            args[0]._fname = f.__name__
            if len(args) > 1:
                args[0]._fargs = args[1:]
            args[0]._fkwargs = kwargs
            if kwargs:
                return f(*args, **kwargs)
            else:
                return f(*args)
        return inner

    @sigwrapper
    def min(self, *args, **kwargs):
        return super(Signal, self).min(*args, **kwargs)

    @sigwrapper
    def transpose(self, *args):
        return super(Signal, self).transpose(*args)
