# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:32:52 2016

@author: drsmith
"""

import numpy as np
import scipy.signal
import scipy.interpolate
from matplotlib import animation
import matplotlib.pyplot as plt

from fdp.classes.fdp_globals import FdpError
from fdp.classes.utilities import isContainer
from . import utilities as UT

def plot2d(*args, **kwargs):
    """Plot 2D signals"""
    return Bes2d(*args, **kwargs)
    

class Bes2d(object):
    """
    """
    
    def __init__(self, container, 
                 tmin=0.0, tmax=5.0,
                 savemovie=False,
                 hightimeres=False):
        
        if not isContainer(container):
            raise FdpError("Use at container level, not signal level")
        self.container = container
        if tmax>10:
            # if tmax large, assume ms input and convert to s
            tmin = tmin/1e3
            tmax = tmax/1e3
        self.tmin = tmin
        self.tmax = tmax
        self.hightimeres = hightimeres
        
        self.signals = None
        self.data = None
        self.time = None
        self.istart = None
        self.istop = None
        
        self.writer = None
        
        self.filter = None
        self.fdata = None
        self.ftime = None
        self.cdata = None
        
        self.getSignals()
        self.setTimeIndices()
        self.loadData()
        self.applyCalibration()
        self.filterData()
        #self.gridData()
        self.makeAnimation()
        if savemovie:
            self.saveAnimationVideo()
        
    def getSignals(self):
        self.signals = UT.get_signals_in_container(self.container)
    
    def setTimeIndices(self):
        time = self.signals[0].time
        self.shot = self.signals[0].shot
        time_indices = np.where(np.logical_and(time>=self.tmin,
                                               time<=self.tmax))[0]
        self.istart = time_indices[0]
        self.istop = time_indices[time_indices.size-1]
        
        self.time = time[self.istart:self.istop+1]
        self.ntime = self.time.size
        print('Data points: {}'.format(self.ntime))

    def loadData(self):        
        nsignals = len(self.signals)
        nrows = nsignals/8
        ntime = self.istop-self.istart+1
        self.data = np.ndarray((nrows,8,ntime))
        for i in range(nrows):
            for j in range(8):
                signal = self.signals[i*8+j]
                zerosignal = np.mean(signal[0:1e3])
                self.data[i,j,:] = signal[self.istart:self.istop+1] - zerosignal
        
    def applyCalibration(self):
        self.calibration = np.mean(self.data[:,:,0:self.ntime/20],axis=2)
        self.rcalibration = np.mean(self.calibration, axis=0)
        #self.calfactor = np.zeros(self.calibration.shape)
        nrows,_ = self.calibration.shape
        for i in range(nrows):
            for j in range(8):
                #self.calfactor[i,j] = self.rcalibration[j]/self.calibration[i,j]
                self.data[i,j,:] = self.data[i,j,:]*self.rcalibration[j]/self.calibration[i,j]
        
    def filterData(self):
        self.filter = scipy.signal.daub(4)
        self.filter = self.filter/np.sum(self.filter)
        self.fdata = scipy.signal.lfilter(self.filter, [1], 
            self.data,
            axis=2)
        #self.fdata = scipy.signal.decimate(self.fdata, 2, axis=2)
        #self.ftime = self.time[::2]
        self.ftime = self.time
        
    def gridData(self):
        nrad = 8
        npol = 4
        rgrid = np.arange(1,nrad+1)
        pgrid = np.arange(1,npol+1)
        rr,pp = np.meshgrid(rgrid, pgrid)
        print(pp.shape)
        rnew = np.arange(0.5,nrad+0.51,0.25)
        pnew = np.arange(0.5,npol+0.51,0.25)
        self.gdata = np.zeros((pnew.size,rnew.size,self.ftime.size))
        print('starting interpolation')
        for i in np.arange(self.ftime.size):
            if i!=0 and np.mod(i+1,100)==0:
                print('  frame {} of {}'.format(i+1,self.ftime.size))
            f = scipy.interpolate.interp2d(rr,
                                           pp,
                                           self.fdata[:,:,i].squeeze(),
                                           kind='linear')
            self.gdata[:,:,i] = f(rnew,pnew)
        
        
    def plotContourf(self, axes=None, index=None):
        return axes.contourf(np.arange(1,8.1),
                             np.arange(1,4.1),
                             self.fdata[:,:,index],
                             cmap=plt.cm.YlGnBu)
                           
    def plotPColorMesh(self, axes=None, index=None):
        return axes.pcolormesh(self.fdata[:,:,index],
                               cmap=plt.cm.YlGnBu)
        
    def makeAnimation(self):
        ims = []
        if self.hightimeres:
            frameint = 1
        else:
            frameint = 10
        nframes = np.int(self.ftime.size/frameint)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.set_xlabel('Radial channels')
        ax1.set_ylabel('Poloidal channels')
        ax1.set_yticks([1,2,3,4])
        ax1.set_aspect('equal')
        ax2 = fig.add_subplot(2,1,2)
        ax2.set_xlim(np.array([self.tmin,self.tmax])*1e3)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Signal (V)')
        fig.subplots_adjust(hspace=0.38)
        print('starting frame loop with {} frames'.format(nframes))
        for i in np.arange(nframes):
            if i!=0 and np.mod(i+1,100)==0:
                print('  frame {} of {}'.format(i+1,nframes))
            im = self.plotContourf(axes=ax1, index=i*frameint)
            #im = self.plotPColorMesh(axes=ax1, index=i*frameint)
            im.set_clim([0, 8])
            pt = ax2.plot(self.ftime*1e3, self.fdata[3,0,:], 'b',
                          self.ftime*1e3, self.fdata[3,5,:], 'g')
            ax1_title = ax1.annotate('BES | {} | t={:.3f} ms'.format(
                self.shot,
                self.ftime[i*frameint]*1e3),
                xy=(0.5, 1.04), 
                xycoords='axes fraction',
                horizontalalignment ='center',
                size='large')
            ln = ax2.plot(np.ones(2)*self.ftime[i*frameint]*1e3,
                          ax2.get_ylim(), 
                          'r')
            an_core = ax2.annotate('Core',
                                  xy=(self.ftime[0]*1e3+0.03,
                                      self.fdata[3,0,15]*1.1),
                                  color='b')
            an_sol = ax2.annotate('SOL',
                                  xy=(self.ftime[0]*1e3+0.03,
                                      self.fdata[3,5,15]*1.1),
                                  color='g')
            ax2_title = ax2.annotate('BES | {} | t={:.3f} ms'.format(
                self.shot,
                self.ftime[i*frameint]*1e3),
                xy=(0.5, 1.04), 
                xycoords='axes fraction',
                horizontalalignment ='center',
                size='large')
            plt.draw()
            if hasattr(im, 'collections'):
                ims.append(im.collections+
                           [pt[0], pt[1], ln[0], ax1_title, 
                            an_core, an_sol, ax2_title])
            else:
                ims.append([im, pt[0], pt[1], ln[0], ax1_title, 
                            an_core, an_sol, ax2_title])
            
        print('calling ArtistAnimation')
        self.animation = animation.ArtistAnimation(fig, ims, 
                                                   blit=False,
                                                   interval=50,
                                                   repeat=False)
                                                   
    def saveAnimationVideo(self):
        print('calling ArtistAnimation.save()')
        filename = 'Bes2d_{}_{}ms.mp4'.format(
            self.shot,
            np.int(self.tmin*1e3))
        writer = animation.FFMpegWriter(fps=24,
                                        bitrate=800)
        self.animation.save(filename, writer=writer)
