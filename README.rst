.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
==============================

Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments.  FDP streamlines data discovery, analysis methods, and visualization.

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://Fusion-Data-Platform.github.io/

Project objectives
---------------------------------

* Integrate data sources, data, and analysis methods in a single, extensible data object

  * Streamline data access - multiple facilities, diagnostics, and databases
  * Organize data and analysis methods in an intuitive object-oriented framework

* Promote collaborative code development and reduce inefficient code duplication

* Reduce barriers to entry for new students and scientists

* Use free and ubiquitous components

  * Python, Numpy, Matplotlib, Github, etc.
  * Platform-independent: desktop Mac/PC, Linux cluster

Example usage
==============================

Initiate an FDP session::

    >>> import fdp
    >>> nstxu = fdp.nstxu()

List diagnostics::

    >>> dir(nstxu.s141000)
    ['bes', 'chers', 'equilibria', 'filterscopes', 'magnetics', 'mpts', 'mse', 'usxr']

View logbook entries::
    
    >>> nstxu.s141000.logbook()
    
List signals::

    >>> nstxu.s141000.equilibria.efit02.listSignals()
    ['psirz', 'qpsi', 'shot', 'userid', 'wmhd']

Plot a signal::

    >>> nstxu.s141000.mpts.te.plot()

List methods for a signal::

    >>> nstx.s141000.bes.d1ch01.listMethods()
    ['animate', 'fft', 'loadConfig', 'plotfft']

Lead developers
==============================

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics

PPPL cluster support from John Schmitt
