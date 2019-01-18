*****************************************
User Guide
*****************************************

Setup for NSTX-U users
=================================

Prerequisites for NSTX-U data access
--------------------------------------------

PPPL computer account: See https://ppplprod.service-now.com/kb_view.do?sysparm_article=KB0010068#

PPPL network access: If you are connected to the `PPPL-WPA` network at PPPL, then your network access is sufficient.  If you are off-site or if you are on-site but your device is not connected to the `PPPL-WPA` network, then you must connect to PPPL network resources via VPN.

* VPN for Macs: https://ppplprod.service-now.com/kb_view.do?sysparm_article=KB0010073#
* VPN for Windows: https://ppplprod.service-now.com/kb_view.do?sysparm_article=KB0010071#
* VPN for Linux: https://ppplprod.service-now.com/kb_view.do?sysparm_article=KB0010168#

NSTX-U data access permissions: Contact `unixadmin@pppl.gov`.  The NSTX-U logbook access requires a special file named `nstxlogs.sybase_login` in your home directory.  If your home directory on the PPPL Linux cluster does not have the file, then generate the file with this command::

  $ source /p/nstxops/util/setup/mkmdsplusdbfile.csh

If you are accessing NSTX-U data directly from your local machine (not via NoMachine or SSH connection to PPPL Linux cluster), then you must copy `nstxlogs.sybase_login` from your Linux home directory to your local home directory.  The file is very short, so you can manually retype it as a plain text file if desired.


Clone FDP
-----------------

Clone FDP using `git`.  (You may need to load `git` with `module load git`.)  The clone command will create a new directory called `fdp` in your current directory.

::

  $ git clone git@github.com:Fusion-Data-Platform/fdp.git


Start FDP session
------------------------

Load the NSTX-U (nstx/mdsplus_alpha) and Python (v2.7 in /p/fdp/anaconda) environment::

  $ module load /u/drsmith/fdp.module

Finally, add your cloned FDP directory to `PYTHONPATH`. Here is the command for t/csh with the cloned FDP directory at $HOME/fdp::

  $ setenv PYTHONPATH $HOME/fdp:${PYTHONPATH}

You may with to put the `module` and `setenv` commands in a shell script file for convenience.  See, for example, `/u/drmsith/fdp.csh`.



Getting started
=====================

This guide is for people who want to use FDP on the PPPL Linux cluster.  If you wish to contribute to the FDP project as a developer, see the developer guide.

To use FDP on the PPPL Linux cluster, load the module nstx/fdp (you may need to unload other nstx modules)::

    [sunfire06:~] % module load nstx/fdp

    [sunfire06:~] % module list
    Currently Loaded Modulefiles:
    1) torque/2.5.2      5) idl/8.2           9) java/v1.6
    2) moab/5.4.0        6) nstx/treedefs    10) nstx/mdsplus5
    3) ppplcluster/1.1   7) nstx/epics       11) nstx/fdp
    4) freetds/0.91      8) nstx/idldirs

Verify that python points to ``/p/fdp/anaconda/bin/python``::

    [sunfire06:~] % which python
    /p/fdp/anaconda/bin/python

If python does not point to ``/p/fdp/anaconda/bin/python``, then PATH contains to a different python distribution.  In this case, you need to modify PATH so ``/p/fdp/anaconda/bin`` is the first python distribution in PATH.

Finally, you can launch python and import the FDP package::

    [sunfire06:~] % python
    Python 2.7.10 |Anaconda 2.3.0 (64-bit)| (default, Sep 15 2015, 14:50:01)
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    Anaconda is brought to you by Continuum Analytics.
    Please check out: http://continuum.io/thanks and https://anaconda.org
    >>> import fdp
    >>>

See Usage Examples to learn about the capabilities of FDP.

Usage examples
=====================

First, import the FDP module::

    >>> import fdp


Initiate a machine instance
-----------------------------------------

**Define** a NSTX machine instance::

    >>> nstxu = fdp.Nstxu()

Shots are added as referenced.  For instance, without previous reference to 139980, you can enter::

    >>> nstxu.s139980.chers.plot()

**Add shots** to the NSTX instance::

    >>> nstxu.addshot(140000)

or a shotlist::

    >>> nstxu.addshot([141400, 141401, 141402])

or by XP::

    >>> nstxu.addshot(xp=1048)

or by date (string or int YYYYMMDD)::

    >>> nstxu.addshot(date=20100817)

**List shots** presently loaded::

    >>> dir(nstxu)

or::

    >>> nstxu.listshot()

Get a custom **shotlist**::

    >>> my_shotlist = nstxu.get_shotlist(xp=1032)  # returns numpy.ndarray

