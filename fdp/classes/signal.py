# -*- coding: utf-8 -*-
"""
signals.py - module containing Signal class

**Classes**

* Signal - signal class for data objects

Created on Tue Jun 23 2015

@author: hyuh
"""
import sys
if sys.version_info > (3,):
    long = int

import inspect
import types
import numpy as np
from .fdp_globals import FdpError, VERBOSE


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
        if VERBOSE:
            print('      {}.__new__ BEGIN with cls {}'.
                  format(kwargs['_name'], cls))
        # ndarray.view().copy() calls __array_finalize__
        obj = np.asanyarray(input_array).view(cls).copy()
        for key in iter(kwargs):
            setattr(obj, key, kwargs[key])
        return obj

    def __init__(self, **kwargs):
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
            # logic for explicit constructor
            return

        if VERBOSE:
            print('        {}.__array_finalize__: BEGIN'.
                  format(self._name))

        # logic for view casting and slicing/copy
        objaxes = getattr(obj, 'axes', None)
        objslic = getattr(obj, '_slic', None)
        if VERBOSE:
            print('          obj.axes is {}'.format(objaxes))
            print('          obj._slic is {}'.format(objslic))

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
        _deltmpattr = True
        if objdict.get('_debug'):
            _deltmpattr = False

        if objaxes:
            for axis in objaxes:
                if objslic:
                    # slice axis according to _slic
                    obj_axis = getattr(obj, axis)
                    if type(objslic) is slice or type(objslic) is list:
                        # logic for 1D arrays
                        if VERBOSE:
                            print('        {}.__array_finalize__: slicing axis {} with obj._slic'.
                                  format(self._name, axis))
                        setattr(self, axis, obj_axis[objslic])
                    elif type(objslic) is tuple:
                        # logic for multi-dim arrays
                        slic_axis = tuple([objslic[objaxes.index(axisaxis)] for
                                           axisaxis in (obj_axis.axes + [axis])])
                        if VERBOSE:
                            print('        {}.__array_finalize__: slicing axis {} with {}'.
                                  format(self._name, axis, slic_axis))
                        if isinstance(slic_axis[0], (int, long, float, np.generic)):
                            if VERBOSE:
                                print('        {}.__array_finalize__: single-point slice'.
                                      format(self._name))
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
                        for axisaxis in obj_axis.axes:
                            if isinstance(objslic[objaxes.index(axisaxis)], (int, long, float, np.generic)):
                                obj_axis.axes.remove(axisaxis)
                    else:
                        raise FdpError('obj._slic is unexpected type')
                else:
                    # obj._slic is undefined; copy each axis as is
                    if VERBOSE:
                        print('        {}.__array_finalize__: copying axis {} to self'.
                              format(self._name, axis))
                    setattr(self, axis, getattr(obj, axis, None))
            # remove all 'point_axes' keys from 'axes' list
#            point_axes = getattr(self, 'point_axes')
#            if point_axes:
#                axes = getattr(self, 'axes')
#                for pa in point_axes:
#                    axis = pa['axis']
#                    if VERBOSE:
#                        print('        {}.__array_finalize__: Trying to delete {} axis'.
#                              format(self._name, axis))
#                    if axis in axes:
#                        if VERBOSE:
#                            print('removing "{}" from self.axes'.format(axis))
#                        axes.remove(axis)
#                    if hasattr(self, axis):
#                        if VERBOSE:
#                            print('deleting attr "{}" from self'.format(axis))
#                        delattr(self, axis)
#                setattr(self, 'axes', axes)
        # end "if objaxes" block

        # clean-up temp attributes
        def delattrtry(ob, at):
            try:
                delattr(ob, at)
            except:
                pass

        if _deltmpattr:
            delattrtry(self, '_slic')
            delattrtry(self, '_fname')
            delattrtry(self, '_fargs')
            delattrtry(self, '_fkwargs')
            delattrtry(obj, '_slic')
            delattrtry(obj, '_fname')
            delattrtry(obj, '_fargs')
            delattrtry(obj, '_fkwargs')

        if VERBOSE:
            print('        {}.__array_finalize__: END'.
                  format(self._name))

    def __array_wrap__(self, out_arr, context=None):
        if VERBOSE:
            print('      {}.__array_wrap__ BEGIN'.format(self._name))
            #print('Called __array_wrap__:')
            #print('__array_wrap__: self is %s' % type(self))
            #print('__array_wrap__:  arr is %s' % type(out_arr))
            # then just call the parent
            #print('__array_wrap__: context is %s' % context)
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_prepare__(self, out_arr, context=None):
        if VERBOSE:
            print('      {}.__array_prepare__ BEGIN'.format(self._name))
            #print('Called __array_prepare__:')
            #print('__array_prepare__: self is %s' % type(self))
            #print('__array_prepare__:  arr is %s' % type(out_arr))
            # then just call the parent
            #print('__array_prepare__: context is %s' % context)
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __getitem__(self, index):
        '''
        self must be Signal class for this to be called, so therefore
        must have the _slic attribute. The _slic attribute preserves indexing for attributes
        '''

        if VERBOSE:
            print('      {}.__getitem__: BEGIN with index {} and shape {}'.
                  format(self._name, index, self.shape))
            #print('      {}.__getitem__: type(self) is {}'.format(self._name, type(self)))
            #print('      {}.__getitem__: self.ndim is {}'.format(self._name, self.ndim))
            #print('      {}.__getitem__: self.shape is {}'.format(self._name, self.shape))
            #print('      {}.__getitem__: type(index) is {}'.format(self._name, type(index)))

        # This passes index to array_finalize after a new signal obj is created
        # to assign axes
        def parseindex(index, dims):
            # format index to account for single elements and pad with appropriate slices.
            #int2slc=lambda i: slice(-1,-2,-1) if int(i) == -1 else slice(int(i),int(i)+1)
            if VERBOSE:
                print('        {}.__getitem__.parseindex(): BEGIN with index {} and dims {}'.
                      format(self._name, index, dims))
            if isinstance(index, (list, slice, np.ndarray)):
                # index is list, slice, or ndarray
                if VERBOSE:
                    print('        {}.__getitem__.parseindex(): index is list|slice|ndarray'.
                          format(self._name))
                if dims < 2:
                    if VERBOSE:
                        print('        {}.__getitem__.parseindex(): ndim < 2 and returning'.
                              format(self._name))
                    return index
                else:
                    if VERBOSE:
                        print('        {}.__getitem__.parseindex(): ndim >= 2'.
                              format(self._name))
                    newindex = [index]
            elif isinstance(index, (int, long, float, np.generic)):
                if VERBOSE:
                    print('        {}.__getitem__.parseindex(): index is int|long|float|generic'.
                          format(self._name))
                newindex = [int(index)]
            elif isinstance(index, tuple):
                if VERBOSE:
                    print('        {}.__getitem__.parseindex(): index is tuple'.
                          format(self._name))
                newindex = [int(i) if isinstance(i, (int, long, float, np.generic))
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
            if VERBOSE:
                print('        {}.__getitem__.parseindex(): END with newindex {}'.
                      format(self._name, newindex))
            return tuple(newindex)

        slcindex = parseindex(index, self.ndim)
        self._slic = slcindex
        if VERBOSE:
            print('      {}.__getitem__: slcindex is {}'.format(
                self._name, slcindex))

        if self._empty is True:
            self._get_mdsdata()

        if VERBOSE:
            #print('      {}.__getitem__: self.shape is {}'.format(self._name, self.shape))
            print('      {}.__getitem__: CALLING super().__getitem__(slcindex)'.
                  format(self._name))

        # super().__getitem__() calls __array_finalize__()
        return super(Signal, self).__getitem__(slcindex)

    def _get_mdsdata(self):
        if self._empty is True:
            # get MDSplus data
            if VERBOSE:
                print('      {}._get_mdsdata: BEGIN'.format(self._name))
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            if VERBOSE:
                print('      {}._get_mdsdata: END'.format(self._name))
            self._empty = False

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
        if VERBOSE:
            print('      {}.__repr__ BEGIN'.format(self._name))
        if self._empty is True:
            self._get_mdsdata()
        if VERBOSE:
            print('      {}.__repr__ CALLING SUPER()'.format(self._name))
        return super(Signal, self).__repr__()

    def __str__(self):
        if VERBOSE:
            print('      {}.__str__ BEGIN'.format(self._name))
        if self._empty is True:
            self._get_mdsdata()
        if VERBOSE:
            print('      {}.__str__ CALLING SUPER()'.format(self._name))
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
        if VERBOSE:
            print('      {}.__getslice__: BEGIN with start {}/stop {}'.
                  format(self._name, start, stop))
        return self.__getitem__(slice(start, stop))

    def __call__(self, **kwargs):
        if VERBOSE:
            print('      {}.__call__ BEGIN'.format(self._name))
        try:
            slc = [slice(None)] * len(self.axes)
        except TypeError:
            print('No axes present for signal {}.'.format(self._name))
            return None
        for kwarg, values in kwargs.items():
            if kwarg not in self.axes:
                print('      {} is not a valid axis.'.format(kwarg))
                raise TypeError
            axis = self.axes.index(kwarg)
            axis_value = getattr(self, kwarg)
            try:
                axis_inds = [np.abs(value - axis_value[:]).argmin()
                             for value in values]
                slc[axis] = slice(axis_inds[0], axis_inds[1])
            except TypeError:
                axis_ind = np.abs(values - axis_value[:]).argmin()
                #axis_inds = [axis_ind, axis_ind+1]
                slc[axis] = axis_ind
        return self[tuple(slc)]

    def __nonzero__(self):
        return bool(self.size)

    def sigwrapper(f):
        def inner(*args, **kwargs):
            #print("getarg decorator: Function {} arguments were: {}, {}".format(f.__name__,args, kwargs))
            args[0]._fname = f.__name__
            if len(args) > 1:
                args[0]._fargs = args[1:]
            args[0]._fkwargs = kwargs
            return f(*args, **kwargs)
        return inner

    @sigwrapper
    def amin(self, *args, **kwargs):
        args[0]._fname = f.__name__
        args[0]._fkwargs = kwargs
        return super(Signal, self).amin(*args, **kwargs)

    @sigwrapper
    def transpose(self, *args, **kwargs):
        return super(Signal, self).transpose(*args, **kwargs)
