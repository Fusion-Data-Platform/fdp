.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html


*****************************************
User Guide
*****************************************

Getting started
=====================

This guide is for people who want to use FDP on the PPPL Linux cluster.  If you wish to contribute to the FDP project as a developer, see the developer guide.

To use FDP on the PPPL Linux cluster, load the module nstx/FDP (you may need to unload other nstx modules)::

    [sunfire06:~] % module load nstx/FDP

    [sunfire06:~] % module list
    Currently Loaded Modulefiles:
    1) torque/2.5.2      5) idl/8.2           9) java/v1.6
    2) moab/5.4.0        6) nstx/treedefs    10) nstx/mdsplus5
    3) ppplcluster/1.1   7) nstx/epics       11) nstx/FDP
    4) freetds/0.91      8) nstx/idldirs 

Verify that python points to ``/p/FDP/anaconda/bin/python``::

    [sunfire06:~] % which python
    /p/FDP/anaconda/bin/python

If python does not point to ``/p/FDP/anaconda/bin/python``, then PATH contains to a different python distribution.  In this case, you need to modify PATH so ``/p/FDP/anaconda/bin`` is the first python distribution in PATH.

Finally, you can launch python and import the FDP package::

    [sunfire06:~] % python
    Python 2.7.10 |Anaconda 2.3.0 (64-bit)| (default, Sep 15 2015, 14:50:01) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    Anaconda is brought to you by Continuum Analytics.
    Please check out: http://continuum.io/thanks and https://anaconda.org
    >>> import FDP
    >>>

See Usage Examples to learn about the capabilities of FDP.

Usage examples
=====================

First, import the FDP module::

    >>> import fdp


Initiate a machine instance
-----------------------------------------

**Define** a NSTX machine instance::

    >>> nstx = fdp.Machine('nstx')

or pre-load a shotlist::

    >>> nstx = fdp.Machine('nstx', [140000, 140001])

or pre-load an XP::

    >>> nstx = fdp.Machine('nstx', xp=1013)


Load shots and XPs
-----------------------------------------

Shots are added as referenced.  For instance, without previous reference to 139980, you can enter::

    >>> nstx.s139980.chers.plot()

**Add shots** to the NSTX instance::

    >>> nstx.addshot(140000)

or a shotlist::

    >>> nstx.addshot([141400, 141401, 141402])

or by XP::

    >>> nstx.addshot(xp=1048)

or by date (string or int YYYYMMDD)::

    >>> nstx.addshot(date=20100817)

**List shots** presently loaded::

    >>> dir(nstx)

or::

    >>> nstx.listshot()

Get a custom **shotlist**::

    >>> my_shotlist = nstx.get_shotlist(xp=1032)  # returns numpy.ndarray

