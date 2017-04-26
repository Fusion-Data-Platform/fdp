========================
How to contribute
========================

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Bugs and feature requests: https://github.com/Fusion-Data-Platform/fdp/issues
* Google group: https://groups.google.com/forum/#!forum/fusion-data-platform

**Workflow**: Thank you for your interest in the FDP project.  To contribute to the FDP code base, please follow the "forking workflow":

#. Fork the main FDP repo at Github to create your personal repo
#. Clone your personal repo to your local work area
#. (Regularly pull updates from the main FDP repo to your local repo)
#. Push commits to your personal repo at Github
#. Submit pull requests to the main FDP repo at Github

Read more:

* https://guides.github.com/activities/forking/
* https://help.github.com/articles/fork-a-repo/

**Style**: Try to follow the `PEP8 style guide <https://www.python.org/dev/peps/pep-0008/>`_.  FDP uses ``flake8`` to scan for PEP8 conformance and static code analysis.  To perform the scans, run ``make lint`` in the top-level directory.

**Testing**: FDP uses ``pytest`` for testing and ``coverage`` for test coverage.  You can run the test suite with ``make test`` or ``pytest`` in the top-level directory, and you can run the test coverage scan with ``make coverage`` in the top-lelvel directory.  See ``test/README.rst`` for more information about testing in FDP.

**Makefile recipes**: The top-level ``Makefile`` contains several recipes for generating docs, code style/quality reviews, and versioning.  Run ``make help`` to see a summary of recipes:

.. code-block:: shell

  $ make help
  help                 show this help message
  docs                 build HTML and PDF documents
  docs-html            build HTML documents
  docs-pdf             build PDF documents
  test                 run pytest in current Python environment
  coverage             check test coverage and show report in terminal
  coverage-html        check test coverage and show report in browser
  lint                 run flake8 for code quality review
  autopep              run autopep8 to fix minor pep8 violations
  authors              update AUTHORS.txt
  changelog            internal use only
  bumpversion          internal use only
  bump-major           update AUTHORS and CHANGELOG; bump major version and commit
  bump-minor           update AUTHORS and CHANGELOG; bump minor version and commit
  bump-patch           update AUTHORS and CHANGELOG; bump patch version and commit
  clean                remove all build, docs, and Python artifacts
  clean-docs           remove docs/build
  clean-pyc            remove Python file artifacts

