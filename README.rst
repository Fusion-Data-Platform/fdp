.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
===========================

.. image:: https://readthedocs.org/projects/fdp/badge/?version=latest
    :target: http://fdp.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments.
FDP streamlines data discovery, management, analysis methods, and visualization.

* Github: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://fdp.readthedocs.io/
* Google group: https://groups.google.com/forum/#!forum/fusion-data-platform

**Description and features**

* An extensible software layer that unites data access, management, analysis, and visualization
* A hierarchical data object with discoverable data and analysis methods
* Data access tasks (servers, trees, nodes, queries) are handled behind the scenes
* A collaborative development platform for data analysis tools
* Built with popular, open-source packages like Numpy and Matplotlib

**Example usage**

A user guide is available in the online documentation at http://fdp.readthedocs.io/

FDP in action:

.. code-block:: python

    import fdp
    nstxu = fdp.nstxu()
    nstxu.s204551.logbook()
    nstxu.s204551.mpts.te.plot()
    nstxu.s204551.equilibria.efit02.kappa.plot()

``nstxu`` is a hierarchical data object that abstracts the NSTX-U machine with easy access to shots, diagnostics, signals, and data methods.  The typical heirarchy is::

    <machine>.<shot>.<diagnostic container>.<signal>.<method>

Shots and signals are auto-loaded when referenced.  Diagnsotics and signals are discoverable using Python's tab-complete functionality or the built-in function ``dir()``.

**Contributing**

To contribute to the FDP project, see `the docs.<http://fdp.readthedocs.io/en/latest/contributing.html>`_

**Lead developers**

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
