.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
==============================

Fusion Data Platform (FDP) is a data framework written in Python for magnetic fusion experiments.  FDP streamlines data discovery, operations, and visualization.

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://Fusion-Data-Platform.github.io/
* Google Group: https://groups.google.com/forum/#!forum/fusion-data-platform

Project objectives
---------------------------------

* Integrate data sources, data, and analysis methods in an extensible framework

  * Streamline data access from multiple facilities, diagnostics, and databases
  * Organize data and analysis methods in an intuitive object-oriented framework

* Promote collaborative code development and reduce inefficient code duplication

  * Users can extend FDP with new capabilities and contribute to the code base

* Reduce barriers to entry for new students and scientists

  * Eliminate the need to learn data access protocols - especially helpful for short-term students and visiting collaborators

* Boost data usage and increase the scientific return-on-investment

  * Obtaining data is expensive - let's make the most of it

* Use free and ubiquitous components

  * Python, Numpy, Matplotlib, Github, etc.
  * Platform-independent: desktop Mac/PC, Linux cluster

Lead developers
---------------------------------

* John Schmitt, Princeton Plasma Physics Lab
* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics


Quick start
================

On the PPPL computing cluster, load the FDP module and start Python (you may need to unload conflicting modules)::

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

