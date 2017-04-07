.DEFAULT_GOAL := help


define LEAD_AUTHORS
Lead developers:
    David R. Smith
    Kevin Tritz
    Howard Yuh
endef
export LEAD_AUTHORS


define FDF_SHORTLOG
Commits from obsolete FDF repository:
   261  David R. Smith
    41  Kevin Tritz
    17  ktritz
    10  Howard Yuh
     7  John Schmitt
     4  hyyuh
     2  jcschmitt
endef
export FDF_SHORTLOG

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"


.PHONY: help
help: ## show this help message
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


.PHONY: docs
docs: docs-html docs-pdf ## build HTML and PDF documents


.PHONY: docs-html
docs-html: ## build HTML documents
	$(MAKE) -C docs/ html
	@$(BROWSER) docs/build/html/index.html


.PHONY: docs-pdf
docs-pdf: ## build PDF documents
	$(MAKE) -C docs/ latexpdf


.PHONY: lint
lint:  ## run flake8 for code quality review
	flake8 --exit-zero fdp/ test/


.PHONY: autopep
autopep:  ## run autopep8 to fix minor pep8 violations
	autopep8 --in-place -r fdp/
	autopep8 --in-place -r test/


.PHONY: authors
authors:  ## create AUTHORS.txt
	@echo "$$LEAD_AUTHORS" > AUTHORS.txt
	@echo "Commits from authors:" >> AUTHORS.txt
	@git shortlog -s -n >> AUTHORS.txt
	@echo "$$FDF_SHORTLOG" >> AUTHORS.txt


.PHONY: bump-major
bump-major: ## bump major version and push new tag
	bumpversion major # runs 'git commit' and 'git tag
	git push --tags


.PHONY: bump-minor
bump-minor: ## bump minor version and push new tag
	bumpversion minor # runs 'git commit' and 'git tag
	git push --tags


.PHONY: bump-patch
bump-patch: ## bump patch version and push new tag
	@cp CHANGELOG.txt CHANGELOG.copy.txt
	@git rm CHANGELOG.txt
	git log --oneline `git describe --tags --abbrev=0`..HEAD > CHANGELOG.txt
	git add -A
	git commit -m "updated CHANGELOG.txt"
	bumpversion --dry-run --list patch | grep -e '[^=]*=\K'
	#@git push --tags
	@rm -f CHANGELOG.copy.txt


.PHONY: clean
clean: clean-pyc clean-docs ## remove all build, docs, and Python artifacts


.PHONY: clean-docs
clean-docs: ## remove docs/build
	rm -rf docs/build


.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
