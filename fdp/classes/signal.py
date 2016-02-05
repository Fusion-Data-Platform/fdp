# -*- coding: utf-8 -*-
"""
signals.py - module containing Signal class

**Classes**

* Signal - signal class for data objects

Created on Tue Jun 23 2015

@author: hyuh
"""

import numpy as np
import inspect
import types
from . import fdp_globals

MDS_SERVERS = fdp_globals.MDS_SERVERS
# implemented MDS_SERVERS from fdf_globals in place of hard-coded MDS server - DRS 10/18/15
FdpError = fdp_globals.FdpError


# commented out and replaced with FdfError - DRS 10/18/15
#class MdsError(Exception):
#    pass

class Signal(np.ndarray):
    """
    sig=fdf.Signal(signal_ndarray, units='m/s', axes=['radius','time'], axes_values=[ax1_1Darray, ax2_1Darray], axes_units=['s','cm']

    e.g.:
    mds.Signal(np.arange((20*10)).reshape((10,20)), units='keV', axes=['radius','time'], axes_values=[100+np.arange(10)*5, np.arange(20)*0.1], axes_units=['s','cm'])

    or an empty signal:
    s=mds.Signal()
    default axes order=[time, space]
    sig=fdf.Signal(units='m/s', axes=['radius','time'], axes_values=[radiusSignal, timeSignal])
    """
    def __init__(self, **kwargs):
        pass

    def __new__(cls, input_array=[], verbose=False, **kwargs):
        #maybe an **kwargs dict for more attr
        #name is name of signal...e.g. Te
        #__doc__ for the signal...decriptor (filled in when? XML/MDSvalue?)
        if verbose:
            print('Called __new__:')
        obj = np.asanyarray(input_array).view(cls).copy()
        #arr = np.asanyarray(input_array).view(cls)
        #print '__new__: type(arr) %s' % type(arr)
        #obj = np.array(arr,copy=True)
        #obj = np.ndarray.__new__(cls, shape=arr.shape, buffer=arr,dtype=arr.dtype)
        #obj = np.ndarray.__new__(cls, shape=arr.shape, dtype=arr.dtype)
        #if input_array is not None:
        #    obj[:] = input_array
        # obj = np.asarray(input_array).view(cls)
        #arr = np.asarray(input_array)
        #obj = np.ndarray.__new__(cls, shape=arr.shape, buffer=arr, dtype=arr.dtype)

        if verbose:
            print('__new__: type(obj) %s' % type(obj))
            print('__new__: setting attributes')
        #obj.units = units
        #obj.axes = axes
        #obj._parent = parent
        #obj._root = root
        #obj._dim_of = dim_of
        obj._verbose = verbose
        obj._slic = None
        obj._empty = True
        #obj._name = name
        #not necessary but can be defined
        #obj.mdstree = mdstree
        #obj.mdsnode = mdsnode
        #obj.mdsshot = mdsshot
        for key,value in iter(kwargs.items()):
            setattr(obj,key,value)

        #Initiate slic attribute to hold slice index info

#        try:
#            for i, axis in enumerate(axes):
#                print(name, axis)
#                setattr(obj, axis, axes_values[i])
#            for axis, reference in zip(axes, axes_values):
#                setattr(obj, axis, reference)
#        except:
#            pass
        return obj

#    def __array_prepare__(self, context=None):
#        print 'In __array_prepare__:'
#        print '   context is' , context

    def __array_finalize__(self, obj):
#        if self._verbose:
#            print('Called __array_finalize__:')
#            print '__array_finalize__: self is type %s  ' % type(self)
#
#            try:
#                print '__array_finalize__: self has len %s' % len(self)
#            except:
#                print '__array_finalize__: self has undefined len'
#
#            print('__array_finalize__: self hasattr(self,"slic") is', hasattr(self,'slic'))
#            print('__array_finalize__: obj is type ', type(obj))
#            try:
#                print '__array_finalize__: obj has len %s' % len(obj)
#            except:
#                print '__array_finalize__: obj has undefined len'
#            print('__array_finalize__: hasattr(obj,"_slic") is', hasattr(obj,"_slic"))
#            if hasattr(obj,'_slic'):
#                print('__array_finalize__: obj._slic is', obj._slic)

        if obj is None:
            return

        #simple copying over of attributes. Defaults to None so hasattr check skipped
        #if hasattr(obj,'units'):
        self.units = getattr(obj, 'units', None)
        #if hasattr(obj,'axes_units'):
        self.axes_units = getattr(obj, 'axes_units', None)
        self.axes = getattr(obj, 'axes', None)
        self._verbose = getattr(obj, '_verbose', False)
        #self._transpose = getattr(obj, '_transpose', None)
        self._parent = getattr(obj, '_parent', None)
        self._empty = getattr(obj, '_empty', None)
        #import pdb; pdb.set_trace()
        #print(obj.__dict__)
        if hasattr(obj,'axes'):
            if obj.axes is not None:
                for i, axis in enumerate(obj.axes):
                    if hasattr(obj,'_slic'):
                        if self._verbose:
                            print('__array_finalize__: type(obj._slic) is  ', type(obj._slic))
                            print('__array_finalize__: obj._slic is  ',obj._slic)

                        try:
                            #1-D
                            if type(obj._slic) is slice or type(obj._slic) is list:
                                setattr(self,axis,getattr(obj, axis)[obj._slic])
                            #>1-D
                            elif type(obj._slic) is tuple:
                                #if getattr(obj, axis).axes != []:
                                #axes is multidimensional, build correct 
                                _slicaxis=tuple([obj._slic[obj.axes.index(axisaxis)] for
                                                 axisaxis in (getattr(obj, axis).axes + [axis])])
                                if self._verbose:
                                    print('__array_finalize__: Assigning axis  ',axis)
                                    print('__array_finalize__: type(_slicaxis) is  ',type(_slicaxis))
                                    print('__array_finalize__: _slicaxis is  ',_slicaxis)
                                    print('__array_finalize__: axis shape is  ', getattr(obj, axis)[_slicaxis].shape)
                                setattr(self,axis,getattr(obj, axis)[_slicaxis])
                            else:
                                if self._verbose:
                                    print('_slic is neither slice, list, nor tuple type for ',axis)

                            #elif type(obj._slic) is list:
                            #    setattr(self,axis,getattr(obj, axis)[obj._slic])
                        except: #must not have a len(), e.g. int type
                            if self._verbose:
                                print('Exception: Axes parsing for ',axis,' failed')
                            pass
                    else:
                        setattr(self,axis,getattr(obj, axis, None))
        self._slic=None


    def __array_wrap__(self, out_arr, context=None):
        if self._verbose:
            print('Called __array_wrap__:')
            print('__array_wrap__: self is %s' % type(self))
            print('__array_wrap__:  arr is %s' % type(out_arr))
            # then just call the parent
            print(context)
        return np.ndarray.__array_wrap__(self, out_arr, context)


    def __getitem__(self,index):
        '''
        self must be Signal class for this to be called, so therefore
        must have the _slic attribute. The _slic attribute preserves indexing for attributes
        '''
        #This passes index to array_finalize after a new signal obj is created to assign axes
        def parseindex(index, dims):
             #format index to account for single elements and pad with appropriate slices.
             int2slc=lambda i: slice(-1,-2,-1) if int(i) == -1 else slice(int(i),int(i)+1)
             if isinstance(index, (list, slice, np.ndarray)):
                 if dims <= 1: return index
                 else: newindex=[index]
             #elif isinstance(index, (int, long, float, np.generic)): newindex=[int2slc(index)]
             elif isinstance(index, (int, long, float, np.generic)): newindex=[int(index)]
             elif isinstance(index, tuple):
                 #newindex = [int2slc(i) if isinstance(i, (int, long, float, np.generic)) else i for i in index]
                 newindex = [int(i) if isinstance(i, (float, np.generic)) else i for i in index]
             ellipsisbool=[Ellipsis is i for i in newindex]
             if sum(ellipsisbool) > 0:
                 ellipsisindex=ellipsisbool.index(True)
                 slcpadding=([slice(None)]*(dims-len(newindex)+1))
                 newindex=newindex[:ellipsisindex] + slcpadding + newindex[ellipsisindex+1:]
             else:
                 newindex=newindex + ([slice(None)]*(dims-len(newindex)))
             return tuple(newindex)

        if self._verbose:
            print('Called __getitem__:')

        slcindex=parseindex(index, self.ndim)
        self._slic=slcindex
        
        #Get the data
        if self._empty is True:
#            try:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty = False
#            except:
#                print 'Something went wrong with getting data'
        # Exec userfunc if method defined:
        if self._verbose:
            print('__getitem__: index is ', index)
            print('__getitem__: type(self._slic) is ', type(self._slic))
            print('__getitem__: self._slic is ', self._slic)
            # print '__getitem__: new is type %s' % type(new)
            print('__getitem__: self is type %s' % type(self))
            # print('__getitem__: self has len %s ' % len(self))
        return super(Signal, self).__getitem__(slcindex)

    def __getattr__(self, attribute):
        if attribute is '_parent':
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        if self._parent is None:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), attribute))
        attr = getattr(self._parent, attribute)
        if inspect.ismethod(attr):
            return types.MethodType(attr.im_func, self)
        else:
            return attr

    def __repr__(self):
        if self._verbose:
            print('Called custom __repr__')
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty = False
        return super(Signal, self).__repr__()
        # return np.asarray(self).__repr__()

    def __str__(self):
        if self._verbose:
            print('Called custom __str__')
        if self._empty is True:
            data = self._root._get_mdsdata(self)
            self.resize(data.shape, refcheck=False)
            self[:] = data
            self._empty = False
        return super(Signal, self).__str__()
        # return np.asarray(self).__str__()

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
            except TypeError:
                axis_ind = np.abs(values-axis_value[:]).argmin()
                axis_inds = [axis_ind, axis_ind+1]
            slc[axis] = slice(axis_inds[0], axis_inds[1])
        return self[tuple(slc)]

    def __nonzero__(self):
        return True
