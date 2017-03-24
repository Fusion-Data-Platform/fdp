.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
==============================

Fusion Data Platform (FDP) is a Python package and data framework for magnetic fusion experiments.  FDP provides an extensible software layer that unites data access, management, analysis, and visualization.

* Documentation: http://Fusion-Data-Platform.github.io/
* Github repository: https://github.com/Fusion-Data-Platform/fdp

Example usage
------------------------------

    >>> nstxu.s204551.mpts.te.plot()
    >>> nstxu.s204620.equilibria.efit02.kappa.plot()
    >>> nstxu.s204670.bes.ch01.plotfft()


Features
------------------------------

* Reduce

FDP reduces barriers to entry for new users and facilitates collaborative development of analysis and visualization tools.  FDP is build on the popular and free platform of Python, Numpy, and Matplotlib, and FDP is collaboratively developed as an open-source project on Github.

Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments.  FDP streamlines data discovery, analysis methods, and visualization.


* Integrate data sources, data, and analysis methods in a single, extensible data object

  * Streamline data access for multiple facilities, diagnostics, and databases
  * Organize data and analysis methods in an intuitive object-oriented framework

* Promote collaborative code development and reduce inefficient code duplication

* Reduce barriers to entry for new students and scientists

* Free components and flexible usage

  * Python, Numpy, Matplotlib, Github, etc.
  * Platform-independent: desktop Mac/PC, Linux cluster
  * Command-line tool or import into code


Lead developers
------------------------------

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics
