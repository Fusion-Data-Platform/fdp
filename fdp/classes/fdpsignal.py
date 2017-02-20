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
from . import fdp_globals

MDS_SERVERS = fdp_globals.MDS_SERVERS
FdpError = fdp_globals.FdpError


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
    def __init__(self, **kwargs):
        pass

    def __new__(cls, input_array=[], verbose=False, **kwargs):
        obj = np.asanyarray(input_array).view(cls).copy()
        if 'Axis' in str(cls):
            obj._verbose = True
        else:
            obj._verbose = verbose
        if obj._verbose:
            print('__new__: cls {}'.format(cls))
            print('__new__: type(cls) {}'.format(type(cls)))
            print('__new__: type(obj) {}'.format(type(obj)))
        obj._empty = True
        for key, value in kwargs.iteritems():
            setattr(obj, key, value)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        objaxes = getattr(obj, 'axes', None)
        objdict = getattr(obj, '__dict__', None)
        _deltmpattr = True

        if self._verbose:
            print('{}.__arrayfinalize__: BEGIN'.format(self._name))

        if objdict is not None:
            if objaxes is not None:
                for key,val in objdict.iteritems():
                    if key not in objaxes:
                        setattr(self, key, val)
            else:
                for key,val in objdict.iteritems():
                    setattr(self, key, val)

            if self._verbose:
                try:
                    print("{}.__arrayfinalize__:Function name {}".
                          format(self._name, obj._fname))
                except AttributeError:
                    pass
                try:
                    print("{}.__arrayfinalize__:Function args {}".
                          format(self._name, obj._fargs))
                except AttributeError:
                    pass
                try:
                    print("{}.__arrayfinalize__:Function kwargs {}".
                          format(self._name, obj._fkwargs))
                except AttributeError:
                    pass

            if '_fname' in objdict and objdict['_fname'] == 'transpose':
                if objaxes is not None:
                    self.axes = [obj.axes[i] for i in objdict['_fargs'][0]] if objdict.has_key('_fargs') else obj.axes[::-1]

            _deltmpattr = True
            if '_debug' in objdict and objdict['_debug']:
                _deltmpattr=False

        if objaxes is not None:
            for axis in objaxes:
                if hasattr(obj,'_slic'): #slice axis according to _slic
                    # reverted to hasattr() to avoid FutureWarning
                    if self._verbose:
                        print('{}.__arrayfinalize__: type(obj._slic) is {}'.
                              format(self._name, type(obj._slic)))
                    try:
                        #1-D
                        if type(obj._slic) is slice or type(obj._slic) is list:
                            setattr(self,axis,getattr(obj, axis)[obj._slic])
                        #>1-D
                        elif type(obj._slic) is tuple:
                            _slicaxis=tuple([obj._slic[obj.axes.index(axisaxis)] for
                                             axisaxis in (getattr(obj, axis).axes + [axis])])
                            if self._verbose:
                                print('{}.__arrayfinalize__: Assigning axis {}'.
                                      format(self._name, axis))
                                print('{}.__arrayfinalize__: type(_slicaxis) is {}'.
                                      format(self._name, type(_slicaxis)))
                                print('{}.__arrayfinalize__: _slicaxis is {}'.
                                      format(self._name, _slicaxis))
                                print('{}.__arrayfinalize__: axis shape is {}'.
                                      format(self._name, getattr(obj, axis)[_slicaxis].shape))
                            setattr(self,axis,getattr(obj, axis)[_slicaxis])
                            if self._verbose:
                                print('{}.__arrayfinalize__: Fixing {} axes'.
                                      format(self._name, axis))
                            for axisaxis in getattr(obj, axis).axes:
                                if isinstance(obj._slic[obj.axes.index(axisaxis)], (int, long, float, np.generic)):
                                    if self._verbose:
                                        print('{}.__arrayfinalize__: Removing {} axis from {}'.
                                              format(self._name, axisaxis,axis))
                                    self.axis.axes.remove(axisaxis)
                                else:
                                    if self._verbose:
                                        print('{}.__arrayfinalize__: {} is not primitive'.
                                              format(type(obj._slic[obj.axes.index(axisaxis)])))
                        else:
                            if self._verbose:
                                print('_slic is neither slice, list, nor tuple type for ',axis)
                    except: #must not have a len(), e.g. int type
                            if self._verbose:
                                print('Exception: Axes parsing for ',axis,' failed')
                    pass
                else: #no slicing, copy each axis as is
                    setattr(self, axis, getattr(obj, axis, None))
                    if self._verbose:
                        print('{}.__arrayfinalize__: no attr _slic in object {}'.
                              format(self._name, obj))

        #clean-up temp attributes
        def delattrtry(ob,at):
            try:
                delattr(ob,at)
            except:
                pass

        if _deltmpattr:
            delattrtry(self,'_slic')
            delattrtry(self,'_fname')
            delattrtry(self,'_fargs')
            delattrtry(self,'_fkwargs')
            delattrtry(obj,'_slic')
            delattrtry(obj,'_fname')
            delattrtry(obj,'_fargs')
            delattrtry(obj,'_fkwargs')

        if self._verbose:
            print('{}.__arrayfinalize__: END'.format(self._name))

    def __array_wrap__(self, out_arr, context=None):
        if self._verbose:
            print('Called __array_wrap__:')
            print('__array_wrap__: self is %s' % type(self))
            print('__array_wrap__:  arr is %s' % type(out_arr))
            # then just call the parent
            print('__array_wrap__: context is %s' % context)
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_prepare__(self, out_arr, context=None):
        if self._verbose:
            print('Called __array_prepare__:')
            print('__array_prepare__: self is %s' % type(self))
            print('__array_prepare__:  arr is %s' % type(out_arr))
            # then just call the parent
            print('__array_prepare__: context is %s' % context)
        return np.ndarray.__array_prepare__(self, out_arr, context)


    def __getitem__(self,index):
        '''
        self must be Signal class for this to be called, so therefore
        must have the _slic attribute. The _slic attribute preserves indexing for attributes
        '''

        if self._verbose:
            print('{}.__getitem__: BEGIN'.format(self._name))
            print('{}.__getitem__: type(self) is {}'.format(self._name, type(self)))
            print('{}.__getitem__: self.ndim is {}'.format(self._name, self.ndim))
            print('{}.__getitem__: self.shape is {}'.format(self._name, self.shape))
            print('{}.__getitem__: type(index) is {}'.format(self._name, type(index)))

        #This passes index to array_finalize after a new signal obj is created to assign axes
        def parseindex(index, dims):
             #format index to account for single elements and pad with appropriate slices.
             #int2slc=lambda i: slice(-1,-2,-1) if int(i) == -1 else slice(int(i),int(i)+1)
             if self._verbose:
                 print('{}.__getitem__: begin parseindex'.format(self._name))
             if isinstance(index, (list, slice, np.ndarray)):
                 # index is list, slice, or ndarray
                 if self._verbose:
                     print('{}.__getitem__: index is list|slice|ndarray'.format(self._name))
                 if dims <= 1:
                     if self._verbose:
                         print('{}.__getitem__: ndim <= 1'.format(self._name))
                     return index
                 else:
                     if self._verbose:
                         print('{}.__getitem__: ndim > 1'.format(self._name))
                     newindex=[index]
             elif isinstance(index, (int, long, float, np.generic)):
                 if self._verbose:
                     print('{}.__getitem__: index is int|long|float|generic'.format(self._name))
                 newindex = [int(index)]
             elif isinstance(index, tuple):
                 if self._verbose:
                     print('{}.__getitem__: index is tuple'.format(self._name))
                 newindex = [int(i) if isinstance(i, (int, long, float, np.generic))
                     else i for i in index]
             # check for ellipses in newindex
             ellipsisbool=[Ellipsis is i for i in newindex]
             if sum(ellipsisbool) > 0:
                 # elipses exists
                 ellipsisindex = ellipsisbool.index(True)
                 slcpadding = ([slice(None)]*(dims-len(newindex)+1))
                 newindex = newindex[:ellipsisindex] \
                     + slcpadding \
                     + newindex[ellipsisindex+1:]
             else:
                 # no elipses
                 newindex = newindex + ([slice(None)]*(dims-len(newindex)))
             if self._verbose:
                 print('{}.__getitem__: end parseindex'.format(self._name))
             return tuple(newindex)

        slcindex = parseindex(index, self.ndim)
        self._slic = slcindex
        if self._verbose:
            print('{}.__getitem__: type(slcindex) is {}'.format(self._name, type(slcindex)))

        if self._empty is True:
            # get MDSplus data
            self._empty=False
            print('{}.__getitem__: getting MDS data'.format(self._name))
            data = self._root._get_mdsdata(self)
            print('{}.__getitem__: resize'.format(self._name))
            self.resize(data.shape, refcheck=False)
            print('{}.__getitem__: attaching MDS data'.format(self._name))
            self[:] = data
            print('{}.__getitem__: end attaching MDS data'.format(self._name))

        if self._verbose:
            print('{}.__getitem__: self.shape is {}'.format(self._name, self.shape))
            print('{}.__getitem__: CALLING super(Signal,self).__getitem__(slcindex)'.
                  format(self._name))

        retvalue = super(Signal,self).__getitem__(slcindex)

        if self._verbose:
            print('{}.__getitem__: END CALL to super(Signal,self).__getitem__(slcindex)'.
                  format(self._name))

        if self._verbose:
            #print('{}: type(self._slic) is ', type(self._slic))
            #print('{}: self._slic is ', self._slic)
            #print '  {}: new is type %s' % type(new)
            #print('{}: self has len %s ' % len(self))
            #print('{}.__getitem__: type(<return>) is {}'.format(self._name, type(retvalue)))
            #print('{}: <return>._name is {}'.format(self._name, retvalue._name))
            #print('{}.__getitem__: <return>.shape is {}'.format(self._name, retvalue.shape))
            print('{}.__getitem__: END'.format(self._name))

        return retvalue


    def __getattr__(self, attribute):
        if attribute is '_parent':
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        if self._parent is None:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        attr = getattr(self._parent, attribute)
        if inspect.ismethod(attr):
            return types.MethodType(attr.__func__, self)
        else:
            return attr

    def __repr__(self):
#        if self._verbose:
#            print('Called custom __repr__')
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty=False
        return super(Signal,self).__repr__()
        #return np.asarray(self).__repr__()

    def __str__(self):
#        if self._verbose:
#            print('Called custom __str__')
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty=False
        return super(Signal,self).__str__()
        #return np.asarray(self).__str__()

    def __getslice__(self, start, stop):
        if self._verbose:
            print('Called __getslice__:')
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
        try:
            slc = [slice(None)] * len(self.axes)
        except TypeError:
            print('No axes present for signal {}.'.format(self._name))
            return None
        for kwarg, values in kwargs.items():
            if kwarg not in self.axes:
                print('{} is not a valid axis.'.format(kwarg))
                raise TypeError
            axis = self.axes.index(kwarg)
            axis_value = getattr(self, kwarg)
            try:
                axis_inds = [np.abs(value-axis_value[:]).argmin()
                             for value in values]
                slc[axis] = slice(axis_inds[0], axis_inds[1])
            except TypeError:
                axis_ind = np.abs(values-axis_value[:]).argmin()
                #axis_inds = [axis_ind, axis_ind+1]
                slc[axis] = axis_ind
        return self[tuple(slc)]

    def __nonzero__(self):
        return True

    def sigwrapper(f):
        def inner(*args, **kwargs):
            #print("getarg decorator: Function {} arguments were: {}, {}".format(f.__name__,args, kwargs))
            args[0]._fname=f.__name__
            if len(args)>1: args[0]._fargs=args[1:]
            args[0]._fkwargs=kwargs
            return f(*args, **kwargs)
        return inner

    @sigwrapper
    def amin(self, *args, **kwargs):
        args[0]._fname=f.__name__
        args[0]._fkwargs=kwargs
        return super(Signal,self).amin(*args, **kwargs)

    @sigwrapper
    def transpose(self, *args, **kwargs):
        return super(Signal,self).transpose(*args, **kwargs)


