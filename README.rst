.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

==============================
Fusion Data Platform
==============================


Fusion Data Platform (FDP) is a data framework for magnetic fusion experiments.  FDP streamlines data discovery, access, management, and visualization.

Resources
===========

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://fusion-data-platform.readthedocs.org/
* Google Group: https://groups.google.com/forum/#!forum/fusion-data-platform

Example
==========
::

    In [1]: import fdp
    
    In [2]: dir(fdp.nstx.s140000)
    Precaching MDS server connections...
    Finished.
    Out[2]: 
    ['bes',
     'chers',
     'equilibria',
     'filterscopes',
     'ip',
     'magnetics',
     'mpts',
     'mse',
     'usxr',
     'vloop']
    
    In [3]: dir(fdp.nstx.s140000.mpts)
    Out[3]: ['comment', 'ne', 'shot', 'spline', 'te', 'valid']
    
    In [4]: fdp.nstx.s140000.mpts.te.plot()
    In [5]: fdp.nstx.s140000.logbook()
    
    Logbook entries for 140000
    ************************************
    140000 on 2010-08-17 in XP 1048
    adiallo in topic PHYS OPS
    
    Small increase of the SPA current by 50 A.
    Good.
    ************************************

Created by
============

* John Schmitt, Princeton Plasma Physics Lab
* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
