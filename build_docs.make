DOCDIR = docs

all:
	rm -rf $(DOCDIR)/build/html/
	$(MAKE) -C $(DOCDIR) html
	git add -A
	##git commit -am "rebuilt docs"

