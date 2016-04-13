.PHONY: build-docs

DOCDIR = docs

build-docs:
	rm -rf $(DOCDIR)/build/html/
	$(MAKE) -C $(DOCDIR) html
