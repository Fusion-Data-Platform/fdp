========================
How to contribute
========================

* Github repository: https://github.com/Fusion-Data-Platform/fdp
* Bugs and feature requests: https://github.com/Fusion-Data-Platform/fdp/issues
* Google group: https://groups.google.com/forum/#!forum/fusion-data-platform

**Code**

Thank you for your interest in the FDP project.  To contribute to the FDP code base, fork the repo at Github and submit pull requests with your contributions.  Read more:

* https://guides.github.com/activities/forking/
* https://help.github.com/articles/fork-a-repo/

**Style**

Try to follow the `PEP8 style guide <https://www.python.org/dev/peps/pep-0008/>`_.  To scan for PEP8 conformance, run ``make lint`` in the top-level directory.

**Makefile utilities**

The top-level ``Makefile`` contains several utilities for generating docs, code style/quality reviews, and versioning:

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
    bump-major           update AUTHORS and CHANGELOG; bump major version and tag
    bump-minor           update AUTHORS and CHANGELOG; bump minor version and tag
    bump-patch           update AUTHORS and CHANGELOG; bump patch version and tag
    clean                remove all build, docs, and Python artifacts
    clean-docs           remove docs/build
    clean-pyc            remove Python file artifacts
