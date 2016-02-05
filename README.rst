.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
==============================

Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments.  FDP streamlines data discovery, operations, and visualization.

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://fusion-data-platform.readthedocs.org/
* Google Group: https://groups.google.com/forum/#!forum/fusion-data-platform

Quick start
================

On the PPPL computing cluster, load the FDP module and start Python::

    $ module load nstx/fdp
    $ python

Plot electron temperature from Thomson scattering for NSTX shot 140000::

    >>> import fdp
    >>> fdp.nstx.s140000.mpts.te.plot()

View logbook entries for NSTX shot 140000::
    
    >>> nstx = fdp.nstx
    >>> nstx.s140000.logbook()
    
    Logbook entries for 140000
    ************************************
    140000 on 2010-08-17 in XP 1048
    adiallo in topic PHYS OPS
    
    Small increase of the SPA current by 50 A.
    Good.
    ************************************

List diagnostic containers for NSTX::

    >>> myshot = nstx.s140000
    >>> dir(myshot)
    ['bes', 'chers', 'equilibria', 'filterscopes', 'magnetics', 'mpts', 'mse', 'usxr']

    >>> dir(myshot.equilibria)
    ['efit01', 'efit02', 'shot']

    >>> dir(myshot.equilibria.efit02)
    ['psirz', 'qpsi', 'shot', 'userid', 'wmhd']

For all shots in XP 1013 on NSTX, plot ion temperature from charge-exchange spectroscopy::

    >>> xp1013 = nstx.filter_shots(xp=1013)
    >>> for shot in xp1013:
    ...     shot.chers.ti.plot()

For all NSTX shots on 8/17/2010, plot the low-f, odd-n magnetics signal::

    >>> myday = nstx.filter_shots(date=20100817)
    >>> for shot in myday:
    ...     shot.magnetics.filtered.lowf_oddn.plot()

Lead Developers
==================

* John Schmitt, Princeton Plasma Physics Lab
* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
