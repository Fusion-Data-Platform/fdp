.DEFAULT_GOAL := help

today := $(shell date +%F)

nextmajorversion := $(shell bumpversion \
  --no-commit --no-tag --dry-run --list major --allow-dirty | \
  grep "^new_version=.*$$" | \
  grep -o "[0-9]*\.[0-9]*\.[0-9]*$$")
nextminorversion := $(shell bumpversion \
  --no-commit --no-tag --dry-run --list minor --allow-dirty | \
  grep "^new_version=.*$$" | \
  grep -o "[0-9]*\.[0-9]*\.[0-9]*$$")
nextpatchversion := $(shell bumpversion \
  --no-commit --no-tag --dry-run --list patch --allow-dirty | \
  grep "^new_version=.*$$" | \
  grep -o "[0-9]*\.[0-9]*\.[0-9]*$$")

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
docs: docs-pdf docs-html ## build HTML and PDF documents


.PHONY: docs-html
docs-html: ## build HTML documents
	$(MAKE) -C docs/ html
	@$(BROWSER) docs/build/html/index.html


.PHONY: docs-pdf
docs-pdf: ## build PDF documents
	$(MAKE) -C docs/ latexpdf


.PHONY: outdated
outdated:  # list outdated conda and pip packages
	@echo "outdated conda packages"
	@conda update --dry-run --all
	@echo "outdated pip packages (may be managed by conda)"
	@pip list --outdated


.PHONY: test
test: ## run pytest in current Python environment
	pytest


.PHONY: coverage
coverage: ## check test coverage and show report in terminal
	@rm -f .coverage
	coverage run --module pytest
	coverage report


.PHONY: coverage-html
coverage-html: coverage ## check test coverage and show report in browser
	@rm -fr htmlcov/
	@coverage html
	@$(BROWSER) htmlcov/index.html


.PHONY: lint
lint:  ## run flake8 for code quality review
	@rm -f flake8.output.txt
	flake8 --exit-zero fdp/


.PHONY: autopep
autopep:  ## run autopep8 to fix minor pep8 violations
	autopep8 --in-place -r fdp/
	autopep8 --in-place -r test/


.PHONY: authors
authors:  ## update AUTHORS.txt
	@echo "$$LEAD_AUTHORS" > AUTHORS.txt
	@echo "Commits from authors:" >> AUTHORS.txt
	@git shortlog -s -n >> AUTHORS.txt
	@echo "$$FDF_SHORTLOG" >> AUTHORS.txt


.PHONY: bump-major
bump-major: authors ## update AUTHORS and CHANGELOG; bump major version and tag
	@cp -f CHANGELOG.txt tmp.txt
	@rm -f CHANGELOG.txt
	@echo "Release v$(nextmajorversion) -- $(today)\n" > CHANGELOG.txt
	@git log --oneline `git describe --tags --abbrev=0`..HEAD >> CHANGELOG.txt
	@echo "\n" >> CHANGELOG.txt
	@cat tmp.txt >> CHANGELOG.txt
	@rm -f tmp.txt
	@git add CHANGELOG.txt AUTHORS.txt
	@git commit -m "updated CHANGELOG.txt and AUTHORS.txt"
	@bumpversion major # runs 'git commit' and 'git tag'


.PHONY: bump-minor
bump-minor: authors ## update AUTHORS and CHANGELOG; bump minor version and tag
	@mv CHANGELOG.txt tmp.txt
	@echo "Release v$(nextminorversion) -- $(today)\n" > CHANGELOG.txt
	@git log --oneline `git describe --tags --abbrev=0`..HEAD >> CHANGELOG.txt
	@echo "\n" >> CHANGELOG.txt
	@cat tmp.txt >> CHANGELOG.txt
	@rm -f tmp.txt
	@git add CHANGELOG.txt AUTHORS.txt
	@git commit -m "updated CHANGELOG.txt and AUTHORS.txt"
	@bumpversion minor # runs 'git commit' and 'git tag'


.PHONY: bump-patch
bump-patch: authors ## update AUTHORS and CHANGELOG; bump patch version and tag
	@cp -f CHANGELOG.txt tmp.txt
	@rm -f CHANGELOG.txt
	@echo "Release v$(nextpatchversion) -- $(today)\n" > CHANGELOG.txt
	@git log --oneline `git describe --tags --abbrev=0`..HEAD >> CHANGELOG.txt
	@echo "\n" >> CHANGELOG.txt
	@cat tmp.txt >> CHANGELOG.txt
	@rm -f tmp.txt
	@git add CHANGELOG.txt AUTHORS.txt
	@git commit -m "updated CHANGELOG.txt and AUTHORS.txt"
	@bumpversion patch # runs 'git commit' and 'git tag'


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
