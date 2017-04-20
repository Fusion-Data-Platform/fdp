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
* A descriptive data object that users can query to find data and analysis methods
* Data access tasks (servers, trees, nodes, queries) are handled behind the scenes
* A collaborative development platform for data analysis tools
* Built with popular, open-source packages like Numpy and Matplotlib

**Example usage**

.. code-block:: python

    import fdp
    nstxu = fdp.nstxu()
    nstxu.s204551.logbook()
    nstxu.s204551.mpts.te.plot()
    nstxu.s204551.equilibria.efit02.kappa.plot()

``nstxu`` is a data object that abstracts the NSTX-U device with easy access to shots, diagnostics, signals, and data methods.  The typical heirarchy is::

    <machine>.<shot>.<diagnostic container>.[<possible sub-containers>].<signal>.<method>

Users can discover data containers like ``mpts``, data signals like ``te``, and data methods like ``plot()`` with Python's tab-complete functionality.

**Contributing**

To contribute to the FDP project, see ``CONTRIBUTING.rst`` in the top-level directory or ``Contributing`` in the docs.

**Lead developers**

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
