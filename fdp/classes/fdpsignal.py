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
VERBOSE = fdp_globals.VERBOSE


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

    def __new__(cls, input_array=[], **kwargs):
        if VERBOSE: print('      {}.__new__ BEGIN'.format(kwargs['_name']))
        obj = np.asanyarray(input_array).view(cls).copy()
        obj._empty = True
        obj.point_axes = []
        for key, value in kwargs.iteritems():
            setattr(obj, key, value)
            #if VERBOSE and key is '_name':
            #    print('      {}.__new__: {} {}'.format(cls, key, value))
        if VERBOSE: print('      {}.__new__ END'.format(obj._name))
        return obj

    def __array_finalize__(self, obj):
        # see https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        if VERBOSE:
            print('      __arrayfinalize__: type(self) is {}'.format(type(self)))
            print('      __arrayfinalize__: type(obj) is {}'.format(type(obj)))
            try: print('      __arrayfinalize__: obj._name is {}'.format(obj._name))
            except: pass
        if obj is None:
            # obj is None betcause __array_finalize__
            # was called from explicit constructor
            return
        else:
            # obj is from view-cast object or template object
            pass

        objdict = getattr(obj, '__dict__', None)
        objaxes = getattr(obj, 'axes', None)
        _deltmpattr = True

        if objdict is not None:
            if objaxes is not None:
                for key,val in objdict.iteritems():
                    if key not in objaxes:
                        setattr(self, key, val)
            else:
                for key,val in objdict.iteritems():
                    setattr(self, key, val)

            if '_fname' in objdict and objdict['_fname'] == 'transpose':
                if objaxes is not None:
                    self.axes = [obj.axes[i] for i in objdict['_fargs'][0]] if objdict.has_key('_fargs') else obj.axes[::-1]

            _deltmpattr = True
            if '_debug' in objdict and objdict['_debug']:
                _deltmpattr=False

        if objaxes is None:
            if VERBOSE: print('      {}.__arrayfinalize__: no axes attribute in obj'.
                              format(self._name))
        else:
            if VERBOSE: print('      {}.__arrayfinalize__: obj contains axes {}'.
                              format(self._name, objaxes))
            for axis in objaxes:
                if VERBOSE: print('      {}.__arrayfinalize__: begin axis {}'.
                                  format(self._name, axis))
                if not hasattr(obj,'_slic'):
                    #no slicing, copy each axis as is
                    if VERBOSE:  print('      {}.__arrayfinalize__: copying axis {} from obj to self'.
                                       format(self._name, axis))
                    setattr(self, axis, getattr(obj, axis, None))
                else:
                    #slice axis according to _slic
                    if VERBOSE:
                        print('      {}.__arrayfinalize__: type(obj._slic) is {}'.
                              format(self._name, type(obj._slic)))
                        print('      {}.__arrayfinalize__: obj._slic is {}'.
                              format(self._name, obj._slic))
                    #try:
                    obj_axis = getattr(obj, axis)
                    if type(obj._slic) is slice or type(obj._slic) is list:
                        setattr(self, axis, obj_axis[obj._slic])
                    elif type(obj._slic) is tuple:
                        _slicaxis=tuple([obj._slic[obj.axes.index(axisaxis)] for
                                         axisaxis in (obj_axis.axes + [axis])])
                        if VERBOSE: print('      {}.__arrayfinalize__: _slicaxis is {}'.
                                          format(self._name, _slicaxis))
                        if isinstance(_slicaxis[0], (int,long,float,np.generic)):
                            if VERBOSE: print('      {}.__arrayfinalize__: single-point slice'.
                                              format(self._name))
                            # "point_axes" is a dict with axis keys and dict values
                            if axis in self.point_axes:
                                raise FdpError('Point axis already present')
                            self.point_axes.append({'axis': axis,
                                                   'value': obj_axis[_slicaxis],
                                                   'units': obj_axis.units})
                        elif isinstance(_slicaxis[0], slice):
                            if VERBOSE: print('      {}.__arrayfinalize__: multi-point slice'.
                                              format(self._name))
                            setattr(self, axis, obj_axis[_slicaxis])
                        else:
                            raise FdpError()
                        for axisaxis in obj_axis.axes:
                            if isinstance(obj._slic[obj.axes.index(axisaxis)], (int, long, float, np.generic)):
                                if VERBOSE: print('      {}.__arrayfinalize__: Removing {} axis from {}'.
                                                  format(self._name, axisaxis,axis))
                                self.axis.axes.remove(axisaxis)
                            else:
                                if VERBOSE: print('      {}.__arrayfinalize__: {} is not primitive'.
                                                  format(self._name,type(obj._slic[obj.axes.index(axisaxis)])))
                    else:
                        raise FdpError()
                    if VERBOSE: print('      {}.__arrayfinalize__: end axis {}'.
                                      format(self._name, axis))

            # remove all 'point_axes' keys from 'axes' list
            point_axes = getattr(self, 'point_axes')
            if point_axes:
                axes = getattr(self, 'axes')
                for pa in point_axes:
                    axis = pa['axis']
                    if axis in axes:
                        axes.remove(axis)
                setattr(self, 'axes', axes)

        # end "if objaxes" block


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

        if VERBOSE:
            print('      {}.__arrayfinalize__ End'.format(self._name))

    def __array_wrap__(self, out_arr, context=None):
        if VERBOSE:
            print('Called __array_wrap__:')
            print('__array_wrap__: self is %s' % type(self))
            print('__array_wrap__:  arr is %s' % type(out_arr))
            # then just call the parent
            print('__array_wrap__: context is %s' % context)
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_prepare__(self, out_arr, context=None):
        if VERBOSE:
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

        if VERBOSE:
            print('      {}.__getitem__: BEGIN'.format(self._name))
            #print('      {}.__getitem__: type(self) is {}'.format(self._name, type(self)))
            #print('      {}.__getitem__: self.ndim is {}'.format(self._name, self.ndim))
            #print('      {}.__getitem__: self.shape is {}'.format(self._name, self.shape))
            #print('      {}.__getitem__: type(index) is {}'.format(self._name, type(index)))

        #This passes index to array_finalize after a new signal obj is created to assign axes
        def parseindex(index, dims):
             #format index to account for single elements and pad with appropriate slices.
             #int2slc=lambda i: slice(-1,-2,-1) if int(i) == -1 else slice(int(i),int(i)+1)
             #if VERBOSE:
             #    print('      {}.__getitem__: begin parseindex'.format(self._name))
             if isinstance(index, (list, slice, np.ndarray)):
                 # index is list, slice, or ndarray
                 if VERBOSE:
                     print('      {}.__getitem__: index is list|slice|ndarray'.format(self._name))
                 if dims <= 1:
                     #if VERBOSE:
                     #    print('      {}.__getitem__: ndim <= 1'.format(self._name))
                     return index
                 else:
                     #if VERBOSE:
                     #    print('      {}.__getitem__: ndim > 1'.format(self._name))
                     newindex=[index]
             elif isinstance(index, (int, long, float, np.generic)):
                 if VERBOSE:
                     print('      {}.__getitem__: index is int|long|float|generic'.format(self._name))
                 newindex = [int(index)]
             elif isinstance(index, tuple):
                 if VERBOSE:
                     print('      {}.__getitem__: index is tuple'.format(self._name))
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
             #if VERBOSE:
             #    print('      {}.__getitem__: end parseindex'.format(self._name))
             return tuple(newindex)

        slcindex = parseindex(index, self.ndim)
        self._slic = slcindex
        if VERBOSE:
            print('      {}.__getitem__: type(slcindex) is {}'.format(self._name, type(slcindex)))
            print('      {}.__getitem__: slcindex is {}'.format(self._name, slcindex))

        if self._empty is True:
            # get MDSplus data
            self._empty=False
            if VERBOSE: print('      {}.__getitem__: getting MDS data'.format(self._name))
            data = self._root._get_mdsdata(self)
            #print('      {}.__getitem__: resize'.format(self._name))
            self.resize(data.shape, refcheck=False)
            if VERBOSE: print('      {}.__getitem__: attaching MDS data'.format(self._name))
            self[:] = data
            if VERBOSE: print('      {}.__getitem__: end attaching MDS data'.format(self._name))

        if VERBOSE:
            print('      {}.__getitem__: self.shape is {}'.format(self._name, self.shape))
            print('      {}.__getitem__: CALLING super().__getitem__(slcindex)'.
                  format(self._name))

        retvalue = super(Signal,self).__getitem__(slcindex)

        if VERBOSE:
            print('      {}.__getitem__: END CALL to super().__getitem__(slcindex)'.
                  format(self._name))
            print('      {}.__getitem__: return shape {} size {}'.
                  format(self._name, retvalue.shape, retvalue.size))

        if VERBOSE:
            #print('      {}: type(self._slic) is ', type(self._slic))
            #print('      {}: self._slic is ', self._slic)
            #print '  {}: new is type %s' % type(new)
            #print('      {}: self has len %s ' % len(self))
            #print('      {}.__getitem__: type(<return>) is {}'.format(self._name, type(retvalue)))
            #print('      {}: <return>._name is {}'.format(self._name, retvalue._name))
            #print('      {}.__getitem__: <return>.shape is {}'.format(self._name, retvalue.shape))
            print('      {}.__getitem__: END'.format(self._name))

        return retvalue


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
#        if VERBOSE:
#            print('Called custom __repr__')
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty=False
        return super(Signal,self).__repr__()
        #return np.asarray(self).__repr__()

    def __str__(self):
#        if VERBOSE:
#            print('Called custom __str__')
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty=False
        return super(Signal,self).__str__()
        #return np.asarray(self).__str__()

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
            print('      {}.__getslice__'.format(self._name))
        return self.__getitem__(slice(start, stop))

    def __call__(self, **kwargs):
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


