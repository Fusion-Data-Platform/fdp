DOCDIR = docs

all:
	cp README.rst LICENSE.rst $(DOCDIR)/source
	rm -rf $(DOCDIR)/build
	#$(MAKE) -C $(DOCDIR) latexpdf
	#cp $(DOCDIR)/build/latex/*.pdf $(DOCDIR)/source/_static
	$(MAKE) -C $(DOCDIR) html

