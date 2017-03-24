.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html

Fusion Data Platform
==============================

Fusion Data Platform (FDP) is a data framework in Python for magnetic fusion experiments.  FDP streamlines data discovery, analysis methods, and visualization.

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Documentation: http://Fusion-Data-Platform.github.io/

Project objectives
==============================

* Integrate data sources, data, and analysis methods in a single, extensible data object

  * Streamline data access for multiple facilities, diagnostics, and databases
  * Organize data and analysis methods in an intuitive object-oriented framework

* Promote collaborative code development and reduce inefficient code duplication

* Reduce barriers to entry for new students and scientists

* Free components and flexible usage

  * Python, Numpy, Matplotlib, Github, etc.
  * Platform-independent: desktop Mac/PC, Linux cluster
  * Command-line tool or import into code

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
    >>> nstxu.s204620.chers.listSignals()
    ['ft', 'nc', 'ti', 'valid', 'vt']

Plot a signal::

    >>> nstxu.s141000.mpts.te.plot()

List methods for a signal::

    >>> nstxu.s141000.bes.d1ch01.listMethods()
    ['animate', 'fft', 'loadConfig', 'plotfft']

Load shots for an XP::

    >>> xp1013 = nstxu.filter_shots(xp=1013)
    >>> dir(xp1013)
    ['s141382', 's141383', 's141384', 's141385', 's141386', 's141387',
    's141388', 's141389', 's141390', 's141391', 's141392', 's141393',
    's141394', 's141395', 's141396', 's141397', 's141398', 's141399',
    's141400', 's141401', 's141402', 's141403', 's141404', 's141405',
    's141406', 's141407', 's141408', 's141409', 's141410', 's141411',
    's141412', 's141413', 's141414']

As the examples above illustrate, the FDP data object is organized like this::

    <machine>.<shot>.<diagnostic>.<signal>.<method>

or, for diagnostic sub-containers like spline profiles and x-ray arrays::

    <machine>.<shot>.<diagnostic>.<sub-container>.<signal>.<method>

Lead developers
==============================

* David R. Smith, U. Wisconsin-Madison
* Kevin Tritz, The Johns Hopkins U.
* Howard Yuh, Nova Photonics