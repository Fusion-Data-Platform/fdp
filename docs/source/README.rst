.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
==============================


Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments.  FDP streamlines data discovery, access, management, and visualization.

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://fusion-data-platform.readthedocs.org/
* Google Group: https://groups.google.com/forum/#!forum/fusion-data-platform

Examples
==========

Plot electron temperature data from Thomson scattering for shot 140000 on NSTX::

    >>> import fdp
    >>> nstx = fdp.nstx
    >>> nstx.s140000.mpts.te.plot()

View logbook entries for NSTX shot 140000::
    
    >>> nstx.s140000.logbook()
    
    Logbook entries for 140000
    ************************************
    140000 on 2010-08-17 in XP 1048
    adiallo in topic PHYS OPS
    
    Small increase of the SPA current by 50 A.
    Good.
    ************************************

List diagnostic containers for NSTX::

    >>> dir(nstx.s140000)
    ['bes', 'chers', 'equilibria', 'filterscopes', 'magnetics', 'mpts', 'mse', 'usxr']

Plot ion temperature data from charge-exchange spectroscopy for all shots in XP 1013 on NSTX::

    >>> nstx.addxp(1013)
    >>> for s in nstx:
    ...     s.chers.ti.plot()

Lead Developers
==================

* John Schmitt, Princeton Plasma Physics Lab
* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
