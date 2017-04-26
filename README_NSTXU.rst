=================================
Setup for NSTX-U users
=================================

Pre-requisites
--------------------

To get access permission for NSTX-U data, contact `unixadmin@pppl.gov`.  NSTX-U logbook access requires a special file named `nstxlogs.sybase_login` in your home directory.  If the file is missing, then generate the file with this command::

  $ source /p/nstxops/util/setup/mkmdsplusdbfile.csh


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
