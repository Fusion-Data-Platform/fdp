# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:20:43 2015

@author: ktritz
"""
import numpy as np
import matplotlib.pyplot as plt


def plot1d(data, xaxis, xlim=None, ylim=None, **kwargs):
    plt.plot(xaxis, data, **kwargs)
    plt.ylabel('{} ({})'.format(data._name, data.units))
    plt.xlabel('{} ({})'.format(xaxis._name, xaxis.units))
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)


def plot2d(data, xaxis, yaxis, **kwargs):
    plot_type = kwargs.pop('type', 'contourf')
    nlevels = int(kwargs.pop('nlevels', 100))
    default_min = float(kwargs.pop('minrange', 0.))
    default_max = float(kwargs.pop('maxrange', 1.))

    plot_func = getattr(plt, plot_type)
    plot_range = set_range(data, default_min, default_max)
    levels = np.linspace(plot_range[0], plot_range[1], nlevels)
    plot_func(np.array(xaxis), np.array(yaxis), np.array(data), levels=levels,
              **kwargs)
    plt.ylabel('{} ({})'.format(yaxis._name, yaxis.units))
    plt.xlabel('{} ({})'.format(xaxis._name, xaxis.units))


def set_range(data, default_min, default_max):
    max_range = data.max()
    min_range = data.min()
    if default_max is 1. and default_min is 0.:
        return min_range, max_range
    hist_data = np.histogram(data, bins=20000)
    cumulative = hist_data[0].cumsum()
    if default_max < 1.0:
        max_index = np.where(cumulative > default_max * cumulative[-1])[0][0]
        max_range = hist_data[1][max_index] * 1.15
    if default_min > 0.0:
        min_index = np.where(cumulative > default_min * cumulative[-1])[0][0]
        min_range = hist_data[1][min_index]
        min_range -= 0.15*abs(min_range)
    return min_range, max_range


def plot3d(data, xaxis, yaxis, zaxis, **kwargs):
    print('3D')


def plot4d(data, xaxis, yaxis, zaxis, taxis, **kwargs):
    print('4D')

plot_methods = [None, plot1d, plot2d, plot3d, plot4d]


def plot(signal, **kwargs):
    if signal._is_container():
        plot_container(signal, **kwargs)
        return
    defaults = getattr(signal, '_plot_defaults', {})
    defaults.update(kwargs)

    signal[:]
    dim_title = None
    point_axes_index = []
    axes_vals = []
    point_axes = np.array([])
    axes_sizes = [getattr(signal, axis).size for axis in signal.axes]
    if 1 in axes_sizes:
        point_axes_index = np.where(np.array(axes_sizes) == 1)[0]
        point_axes = np.array(signal.axes)[point_axes_index]
        axes_vals = [getattr(signal, axis) for axis in point_axes]
        # axes_units = [axis.units for axis in axes_vals]
        dim_title = ', '.join(['{} = {:.3f}'.format(axis, val)
                               for axis, val
                               in zip(point_axes, axes_vals)])
        signal = signal.squeeze()
    dims = signal.ndim
    if kwargs.get('overplot', None) is None:
        plt.figure()
    multi_axis = kwargs.get('multi', None)
    if multi_axis is 'shot':
        plot_multishot(signal, **defaults)
        return
    if multi_axis in signal.axes and dims > 1:
        plot_multi(signal, **defaults)
        return
    axes = [getattr(signal, axis) for axis in
            set(signal.axes).difference(point_axes)]
    plot_methods[dims](signal, *axes, **defaults)
    plt.title(signal._name, fontsize=20)
    plt.suptitle('Shot #{} {}'.format(signal.shot, dim_title), x=0.5, y=1.00,
                 fontsize=20, horizontalalignment='center')


def plot_multi(signal, **kwargs):
    axis_name = kwargs.pop('multi', None)
    plot_type = kwargs.pop('type', 'contourf')
    nlevels = int(kwargs.pop('nlevels', 100))
    default_min = float(kwargs.pop('minrange', 0.))
    default_max = float(kwargs.pop('maxrange', 1.))

    plot_range = set_range(signal, default_min, default_max)
    axes = [getattr(signal, axis) for axis in signal.axes]
    axis_index = signal.axes.index(axis_name)
    multi_axis = axes.pop(axis_index)
    ax = plt.subplot(111)
    ax.grid()
    legend = kwargs.pop('legend', False)
    for index, label in enumerate(multi_axis):
        label = '{} = {:.3f} {}'.format(axis_name, label, multi_axis.units)
        data = np.take(signal, index, axis=axis_index)
        plot_axes = [np.take(axis, index, axis=axis.axes.index(axis_name))
                     if axis_name in axis.axes else axis for axis in axes]
        plot_methods[data.ndim](data, *plot_axes, label=label, **kwargs)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.subplots_adjust(right=0.65)
    plt.show()


def plot_container(container, **kwargs):
    [signal.plot() for signal in container._signals.values()]
