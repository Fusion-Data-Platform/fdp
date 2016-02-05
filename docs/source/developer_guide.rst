.. Restructured Text (RST) Syntax Primer: http://sphinx-doc.org/rest.html



*****************************************
Developer Guide
*****************************************

Getting started
=========================

This guide is for developers who want to contribute to the FDP project, and this guide describes the development workflow on the PPPL Linux cluster.  If you simply want to use FDP on the PPPL Linux cluster, see the user guide.

The FDP code repository is hosted on GitHub: https://github.com/Fusion-Data-Platform/FDP

To participate in the FDP project as a developer, you must create a GitHub account.  The FDP project uses GitHub and Git for collaborative development and version control.

Configure Git
-------------------------

On the PPPL Linux cluster, load the module git/1.8.0.2 (on Red Hat 6 systems, use git/2.4.2)::

    [sunfire08:~] % module avail git
    --------------------- /usr/pppl/Modules/modulefiles -------------------
    git/1.7.4.1(default)     git/1.8.0.2      git/2.4.2
    
    [sunfire08:~] % module load git/1.8.0.2
    
    [sunfire08:~] % module list
    Currently Loaded Modulefiles:
    1) torque/2.5.2      3) ppplcluster/1.1
    2) moab/5.4.0        4) git/1.8.0.2

You may want to add the module load command to your shell start-up files: ~/.cshrc for csh/tcsh or ~/.bash_profile for bash.

Next, you must configure Git with your name and email (the same email associated with your GitHub account)::

    [sunfire08:~] % git config --global user.name "John Doe"
    [sunfire08:~] % git config --global user.email "JohnDoe@email.com"

Also, you may want to set a default editor (e.g. vi, emacs, nedit) for Git comments::

    [sunfire08:~] % git config --global core.editor nedit

You can inspect your Git configuration in the file ~/.gitconfig.  For more information about Git configuration, see https://help.github.com/articles/set-up-git/ or https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup

Clone the FDP repository
-------------------------------

Git clones repositories into a new directory in your current directory.  In the right column of the FDP repo page (https://github.com/Fusion-Data-Framework/FDP), you can find the HTTPS URL (https://github.com/Fusion-Data-Framework/FDP.git) to clone FDP to your local directory ::

    [sunfire08:~] % ls -d FDP
    ls: FDP: No such file or directory
    
    [sunfire08:~] % git clone https://github.com/Fusion-Data-Framework/FDP.git
    Cloning into 'FDP'...
    remote: Counting objects: 619, done.
    remote: Total 619 (delta 0), reused 0 (delta 0), pack-reused 619
    Receiving objects: 100% (619/619), 783.01 KiB, done.
    Resolving deltas: 100% (279/279), done.
    
    [sunfire08:~] % ls -d FDP
    FDP/

Cloning via SSH is also feasible: https://help.github.com/articles/set-up-git/#next-steps-authenticating-with-github-from-git

Finally, add your new FDP directory to the ``PYTHONPATH`` environment variable::

    [sunfire08:~] % setenv PYTHONPATH ${HOME}/FDP:$PYTHONPATH

    [sunfire08:~] % echo $PYTHONPATH
    /u/drsmith/FDP:<other directories>

You may want to add this action to your shell start-up files, as described above.  In bash, use the export command to set ``PYTHONPATH``.


Git workflow for FDP development
--------------------------------------

\(1) Create a development branch (here, we call it devbranch) and checkout the new branch::

    [sunfire08:~] % cd FDP
    
    [sunfire08:~/FDP] % git branch
    * master
    
    [sunfire08:~/FDP] % git branch devbranch
    
    [sunfire08:~/FDP] % git branch
    devbranch
    * master
    
    [sunfire08:~/FDP] % git checkout devbranch
    Switched to branch 'devbranch'
    
    [sunfire08:~/FDP] % git branch
    * devbranch
    master 


Devbranch initializes as a copy of master.  ``git branch`` lists branches in your local repository, and the asterisk denotes the active branch.  You can switch between local branches with ``git checkout <LocalBranchName>``.

\(2) Push devbranch to the remote FDP repository at GitHub (you may need to enter your GitHub username and password)::

    [sunfire08:~/FDP] % git push origin devbranch
    Total 0 (delta 0), reused 0 (delta 0)
    To https://github.com/Fusion-Data-Framework/FDP.git
     * [new branch]      devbranch -> devbranch

devbranch is now listed in the FDP repository at GitHub.  ``origin`` is the alias for the remote GitHub repository.  You can view your remote repositories and aliases with ``git remote -v``.


\(3) Proceed with FDP development within devbranch: commit changes, add/delete files, and push updates to GitHub.

As you complete small tasks, you should commit changes to your local repository with ``git commit -a -m '<mymessage>'``.  Also, each commit requires a short message describing the changes::

    [sunfire02:~/FDP] % git commit -a -m 'added dictionary rows in logbook.py'
    [devbranch bb6c58a] added dictionary rows in logbook.py
    1 file changed, 16 insertions(+), 21 deletions(-) 

If you do not specify a commit message with -m option, then Git will open your default editor and ask for a commit message (see Configure Git above).  The -a option commits all file changes throughout the branch index, not simply your current directory.  The branch index is the list of files Git tracks in the branch.  ``git commit -a`` tracks changes to files in the branch index, so you must add new files to the index and remove deleted files from the index.  You can view the branch index with ``git ls-files``, and you can add new files to the index and remove deleted files from the index with ``git add -A``::

    [sunfire02:~/FDP] % touch temp.py

    [sunfire02:~/FDP] % ls temp.py
    temp.py

    [sunfire02:~/FDP] % git ls-files temp.py

    [sunfire02:~/FDP] % git add -A

    [sunfire02:~/FDP] % git ls-files temp.py
    temp.py 

Note that temp.py appeared in the index only after the command ``git add -A``.  Similarly, deleted files stay in the index until the ``git add -A`` is given.

When you complete a large task, you should “push” changes to the devbranch on GitHub with ``git push``::

    [sunfire05:~/FDP] % git push origin devbranch
    Counting objects: 10, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (6/6), done.
    Writing objects: 100% (6/6), 1.30 KiB, done.
    Total 6 (delta 3), reused 0 (delta 0)
    To https://github.com/Fusion-Data-Framework/FDP.git
        129c5d9..a166825 devbranch -> devbranch

Again, "origin" signifies the branches on the remote GitHub repo.

\(4) While you are working locally in devbranch, others may be modifying master at GitHub.  When you are ready to merge devbranch into master, you should first merge the latest version of master from GitHub into your local devbranch.  To retrieve the latest version of master from GitHub, use ``git fetch``::

    [sunfire05:~/FDP] % git fetch origin master
    From https://github.com/Fusion-Data-Framework/FDP 
    * branch            master     -> FETCH_HEAD

Next, verify that you are in devbranch and merge origin/master into devbranch::

    [sunfire08:~/FDP] % git branch
    * devbranch
    master
    
    [sunfire05:~/FDP] % git merge origin/master 

Next, push your local devbranch to devbranch on GitHub::

    [sunfire05:~/FDP] % git push origin devbranch

Finally, on the GitHub website, in the devbranch area, submit a *pull request* to pull devbranch into master.




Package reference
=============================


Class Fdp
-----------------------
.. autoclass:: fdp.classes.fdp.Fdp
    :members:

Class Machine
-----------------------
.. autoclass:: classes.machine.Machine
    :members:

Class Shot
-----------------------
.. autoclass:: classes.shot.Shot
    :members:

Class Container
-----------------------
.. autoclass:: classes.container.Container
    :members:

Class Signal
-----------------------
.. autoclass:: classes.fdpsignal.Signal
    :members:

Class Logbook
-----------------------
.. autoclass:: classes.logbook.Logbook
    :members:












