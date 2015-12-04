.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html



This guide is for people who want to use FDF on the PPPL Linux cluster.  If you wish to contribute to the FDF project as a developer, see the developer guide.

.. only:: latex
    
    HTML documentation is also available: http://fusion-data-framework.github.io/fdf/

.. only:: html
    
    `PDF Documentation <http://fusion-data-framework.github.io/fdf/_static/FusionDataFramework.pdf>`_ is also available.

To use FDF on the PPPL Linux cluster, load the module nstx/fdf (you may need to unload other nstx modules)::

    [sunfire06:~] % module load nstx/fdf

    [sunfire06:~] % module list
    Currently Loaded Modulefiles:
    1) torque/2.5.2      5) idl/8.2           9) java/v1.6
    2) moab/5.4.0        6) nstx/treedefs    10) nstx/mdsplus5
    3) ppplcluster/1.1   7) nstx/epics       11) nstx/fdf
    4) freetds/0.91      8) nstx/idldirs 

Verify that python points to ``/p/fdf/anaconda/bin/python``::

    [sunfire06:~] % which python
    /p/fdf/anaconda/bin/python

If python does not point to ``/p/fdf/anaconda/bin/python``, then PATH contains to a different python distribution.  In this case, you need to modify PATH so ``/p/fdf/anaconda/bin`` is the first python distribution in PATH.

Finally, you can launch python and import the FDF package::

    [sunfire06:~] % python
    Python 2.7.10 |Anaconda 2.3.0 (64-bit)| (default, Sep 15 2015, 14:50:01) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    Anaconda is brought to you by Continuum Analytics.
    Please check out: http://continuum.io/thanks and https://anaconda.org
    >>> import fdf
    >>>

See Usage Examples to learn about the capabilities of FDF.

