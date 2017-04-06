.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
===========================

Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments. FDP streamlines data discovery, management, analysis methods, and visualization.

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://Fusion-Data-Platform.github.io/
* Google group: https://groups.google.com/forum/#!forum/fusion-data-platform

**Description**

* An extensible software layer that unites data access, management, analysis, and visualization in a single data object
* A descriptive data object that users can query to find data and analysis methods
* A data object that handles data access (servers, trees, nodes, queries) behind the scenes
* A collaborative development plateform for data analysis tools
* Built with popular, open-source packages like Numpy and Matplotlib

**Example usage**

.. code-block:: python

    import fdp
    nstxu = fdp.nstxu()
    nstxu.s204551.mpts.te.plot()
    nstxu.s204620.equilibria.efit02.kappa.plot()
    nstxu.s204670.bes.ch01.plotfft()
    
Python's tab-complete feature presents users with data containers like ``mpts`` and ``efit02``, data signals like ``te`` and ``kappa``, and data methods like ``plot()`` and ``plotfft()``.

**Lead developers**

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
