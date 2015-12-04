#import importlib as _importlib
#import pkgutil as _pkgutil
#__all__ = [_mod[1].split(".")[-1] for _mod in
#           filter(lambda _mod: _mod[1].count(".") == 1 and not
#                               _mod[2] and __name__ in _mod[1],
#                  [_mod for _mod in _pkgutil.walk_packages("." + __name__)])]
#__sub_mods__ = [".".join(_mod[1].split(".")[1:]) for _mod in
#                filter(lambda _mod: _mod[1].count(".") > 1 and not
#                                    _mod[2] and __name__ in _mod[1],
#                       [_mod for _mod in
#                        _pkgutil.walk_packages("." + __name__)])]
#from . import *
#for _module in __sub_mods__:
#    _importlib.import_module("." + _module, package=__name__)
import pkgutil as _pkgutil

__path__ = _pkgutil.extend_path(__path__, __name__)
for _importer, _modname, _ispkg in _pkgutil.walk_packages(path=__path__,
                                                          prefix=__name__+'.'):
      __import__(_modname)