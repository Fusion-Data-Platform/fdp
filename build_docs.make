DOCDIR = docs

all:
	### copy README and LICENSE to docs/source
	git rm $(DOCDIR)/source/README.rst
	git rm $(DOCDIR)/source/LICENSE.rst
	cp README.rst LICENSE.rst $(DOCDIR)/source
	git add -A
	### rebuild docs
	rm -rf $(DOCDIR)/build
	$(MAKE) -C $(DOCDIR) html
	git add -A
	git commit -am "rebuilt docs"

