.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
===========================

Fusion Data Platform (FDP) is a Python package and data framework for magnetic fusion experiments.

* Documentation: http://Fusion-Data-Platform.github.io/
* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Google group: https://groups.google.com/forum/#!forum/fusion-data-platform

**Description**

* An extensible software layer that unites data access, management, analysis, and visualization in a single data object
* A descriptive data object that users can easily inspect to find data and analysis methods
* A data object that handles data access (servers, nodes, trees, tables, etc.) behind the scenes
* A collaborative platform for development of analysis tools
* Intelligent multi-dimensional slicing documents singleton dimensions (e.g. a radial profile knows the time slice to which it corresponds)
* Built with popular, open-source packages like Numpy and Matplotlib

**Example usage**

    >>> import fdp
    >>> nstxu = fdp.nstxu()
    >>> nstxu.s204551.mpts.te.plot()
    >>> nstxu.s204620.equilibria.efit02.kappa.plot()
    >>> nstxu.s204670.bes.ch01.plotfft()
    
Python's tab-complete feature presents users with data containers like ``mpts`` and ``efit02``, data signals like ``te`` and ``kappa``, and data methods like ``plot()`` and ``plotfft()``.

Exercise: Create 2D plots of electron density for every shot in XP 1013::

    import fdp
    nstxu = fdp.nstxu()
    xp1013 = nstxu.filter_shots(xp=1013)
    for shot in xp1013:
        shot.mpts.ne.plot()

**Lead developers**

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
