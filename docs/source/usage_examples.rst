.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html


.. _usage-examples:
*****************************************
Usage Examples
*****************************************

First, import the FDF module::

    >>> import fdf


Initiate a machine instance
=======================================

**Define** a NSTX machine instance::

    >>> nstx = fdf.Machine('nstx')

or pre-load a shotlist::

    >>> nstx = fdf.Machine('nstx', [140000, 140001])

or pre-load an XP::

    >>> nstx = fdf.Machine('nstx', xp=1013)


Load shots and XPs
=======================================

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



