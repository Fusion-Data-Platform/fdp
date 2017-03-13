# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:05:14 2015

@author: ktritz
"""
from collections import Mapping, MutableMapping, deque
import os
import numpy as np
from warnings import warn
import MDSplus as mds
from .logbook import Logbook
from .shot import Shot
from .fdp_globals import FDP_DIR, FdpError, FdpWarning, VERBOSE
from .datasources import machineAlias, MDS_SERVERS, EVENT_SERVERS


class Machine(MutableMapping):
    """
    Factory root class that contains shot objects and MDS access methods.

    **Usage**::

        >>> import fdf
        >>> nstxu = fdf.nstxu()
        >>> nstxu.s140000.logbook()
        >>> nstxu.s140000.mpts.plot()

    """

    # Maintain a dictionary of cached MDS server connections to speed up
    # access for multiple shots and trees. This is a static class variable
    # to avoid proliferation of MDS server connections
    _connections = []
    _parent = None
    _modules = None

    def __init__(self, name='nstxu', shotlist=None, xp=None, date=None):
        self._shots = {}  # shot dictionary with shot number (int) keys
        self._classlist = {}
        self._name = machineAlias(name)
        if VERBOSE: print('{}.__init__'.format(self._name))
        self._logbook = Logbook(name=self._name, root=self)
        self._eventConnection = mds.Connection(EVENT_SERVERS[self._name])
        if len(self._connections) is 0:
            if VERBOSE: print('{}.__init__  Precaching MDS connections...'.
                              format(self._name))
            for _ in range(2):
                try:
                    connection = mds.Connection(MDS_SERVERS[self._name])
                    connection.tree = None
                    self._connections.append(connection)
                except:
                    msg = 'MDSplus connection to {} failed'.format(
                        MDS_SERVERS[self._name])
                    raise FdpError(msg)
            if VERBOSE: print('{}.__init__  Finished MDS'.format(self._name))
        self.s0 = Shot(0, root=self, parent=self)
        if shotlist or xp or date:
            self.addshot(shotlist=shotlist, xp=xp, date=date)

    def __getattr__(self, name):
        if VERBOSE: print('{}.__getattr__({})'.format(self._name, name))
        try:
            shot = int(name.split('s')[1])
        except:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                                 type(self), name))
        if (shot not in self._shots):
            if VERBOSE: print('{}.__getattr__: loading shot {}'.
                              format(self._name, shot))
            self._shots[shot] = Shot(shot, root=self, parent=self)
        return self._shots[shot]

    def __repr__(self):
        return '<machine {}>'.format(self._name.upper())

    def __iter__(self):
        return iter(self._shots.values())

    def __contains__(self, value):
        return value in self._shots

    def __len__(self):
        return len(self._shots.keys())

    def __delitem__(self, item):
        self._shots.__delitem__(item)

    def __getitem__(self, item):
        if VERBOSE: print('{}.__getitem__({})'.format(self._name, item))
        if item == 0:
            return self.s0
        return self._shots[item]

    def __setitem__(self, item, value):
        pass

    def __dir__(self):
        shotlist = ['s0']
        shotlist.extend(['s{}'.format(shot) for shot in self._shots.iterkeys()])
        return shotlist

    def _get_connection(self, shot, tree):
        for connection in self._connections:
            if connection.tree == (tree, shot):
                self._connections.remove(connection)
                self._connections.insert(0, connection)
                return connection
        connection = self._connections.pop()
        try:
            connection.closeAllTrees()
        except:
            pass
        try:
            connection.openTree(tree, shot)
            connection.tree = (tree, shot)
        except:
            connection.tree = (None, None)
        finally:
            self._connections.insert(0, connection)
        return connection

    def _get_mdsdata(self, signal):
        if VERBOSE:
            print('{}._get_mdsdata({}): BEGIN'.
                  format(self._name, signal._name))
        shot = signal.shot
        if shot is 0:
            print('No MDS data exists for model tree')
            return None
        connection = self._get_connection(shot, signal._mdstree)
        try:
            data = connection.get(signal._mdsnode)
        except:
            msg = 'MDSplus connection error for shot {}, tree {}, and node {}'.format(
                signal.shot, signal._mdstree, signal._mdsnode)
            warn(msg, FdpWarning)
            return np.zeros(0)
        try:
            if hasattr(signal, '_raw_of') and signal._raw_of is not None:
                if VERBOSE:
                    print('{}._get_mdsdata({}): trying data.raw_of()'.
                          format(self._name, signal._name))
                data = data.raw_of()
            else:
                if VERBOSE:
                    print('{}._get_mdsdata({}): no data.raw_of()'.
                          format(self._name, signal._name))
        except:
            if VERBOSE:
                print('{}._get_mdsdata({}): threw exception'.
                      format(self._name, signal._name))
        try:
            if hasattr(signal, '_dim_of') and signal._dim_of is not None:
                if VERBOSE:
                    print('{}._get_mdsdata({}): trying data.dim_of()'.
                          format(self._name, signal._name))
                data = data.dim_of()
            else:
                if VERBOSE:
                    print('{}._get_mdsdata({}): no data.dim_of()'.
                          format(self._name, signal._name))
        except:
            if VERBOSE:
                print('{}._get_mdsdata({}): threw exception'.
                      format(self._name, signal._name))
        data = data.value_of().value
        if signal._transpose is not None:
            data = data.transpose(signal._transpose)
        if hasattr(signal, '_postprocess'):
            data = signal._postprocess(data)
        if VERBOSE:
            print('{}._get_mdsdata({}): END with type(data) {}'.
                  format(self._name, signal._name, type(data)))
        return data

    def _get_modules(self):
        if VERBOSE: print('{}._get_modules()'.format(self._name))
        if self._modules is None:
            if VERBOSE: print('{}._get_modules() Surveying diagnostic modules'.
                              format(self._name))
            module_dir = os.path.join(FDP_DIR, 'modules', self._name)
            self._modules = [module for module in os.listdir(module_dir)
                        if os.path.isdir(os.path.join(module_dir, module)) and
                        module[0] is not '_']
        return self._modules

    def addshot(self, shotlist=None, date=None, xp=None, verbose=False):
        """
        Load shots into the Machine class

        **Usage**

            >>> nstxu.addshot([140000 140001])
            >>> nstxu.addshot(xp=1032)
            >>> nstxu.addshot(date=20100817, verbose=True)

        Note: You can reference shots even if the shots have not been loaded.

        """
        if shotlist and not isinstance(shotlist, list):
            shotlist = [shotlist]
        if xp and not isinstance(xp, list):
            xp = [xp]
        if date and not isinstance(date, list):
            date = [date]
        shots = []
        if shotlist:
            shots.extend([shotlist])
        if date or xp:
            shots.extend(self._logbook.get_shotlist(date=date, xp=xp,
                                                    verbose=verbose))
        for shot in np.unique(shots):
            if shot not in self._shots:
                self._shots[shot] = Shot(shot, root=self, parent=self)

    def addxp(self, xp=[]):
        self.addshot(xp=xp)

    def adddate(self, date=[]):
        self.addshot(date=date)

    def listshot(self):
        keys = self._shots.keys()
        keys.sort()
        for shotkey in keys:
            shot = self._shots[shotkey]
            print('{} in XP {} on {}'.format(shot.shot, shot.xp, shot.date))

    def get_shotlist(self, date=[], xp=[], verbose=False):
        # return a list of shots
        return self._logbook.get_shotlist(date=date, xp=xp, verbose=verbose)

    def setevent(self, event, shot_number=None, data=None):
        event_data = bytearray()
        if shot_number is not None:
            shot_data = shot_number // 256**np.arange(4) % 256
            event_data.extend(shot_data.astype(np.ubyte))
        if data is not None:
            event_data.extend(str(data))
        mdsdata = mds.mdsdata.makeData(np.array(event_data))
        event_string = 'setevent("{}", {})'.format(event, mdsdata)
        status = self._eventConnection.get(event_string)
        return status

    def wfevent(self, event, timeout=0):
        event_string = 'kind(_data=wfevent("{}",*,{})) == 0BU ? "timeout"' \
                       ': _data'.format(event, timeout)
        data = self._eventConnection.get(event_string).value
        if type(data) is str:
            raise FdpError('Timeout after {}s in wfevent'.format(timeout))
        if not data.size:
            return None
        if data.size > 3:
            shot_data = data[0:4]
            shot_number = np.sum(shot_data * 256**np.arange(4))
            data = data[4:]
            return shot_number, ''.join(map(chr, data))
        return data

    def find(self, tag, obj=None):
        root = getattr(self, '_root', self)
        find_list = set([])
        for module in root.s0._modules:
            module_obj = getattr(root.s0, module)
            container_queue = deque([module_obj])
            while True:
                try:
                    container = container_queue.popleft()
                    container._get_dynamic_containers()
                    container_queue.extend(container._containers.values())
                    if obj is None or obj.lower() == 'signal':
                        for signal in container._signals.values():
                            if signal._contains(tag):
                                branch_str = '.'.join([signal._get_branch(),
                                                      signal._name])
                                find_list.add(branch_str)
                    if obj is None or obj.lower() == 'axis':
                        for signal in container._signals.values():
                            for axis_str in signal.axes:
                                axis = getattr(signal, axis_str)
                                if axis._contains(tag):
                                    branch_str = '.'.join([signal._get_branch(),
                                                     signal._name, axis._name])
                                    find_list.add(branch_str)
                    if obj is None or obj.lower() == 'container':
                        if container._contains(tag):
                            find_list.add(container._get_branch())
                except IndexError:
                    break
        find_list = list(find_list)
        find_list.sort()
        return find_list

    def filter_shots(self, date=[], xp=[]):
        """
        Get a Machine-like object with an immutable shotlist for XP(s)
        or date(s)
        """
        self.addshot(xp=xp, date=date)
        return ImmutableMachine(xp=xp, date=date, parent=self)


class ImmutableMachine(Mapping):
    """
    An immutable Machine-like class for dates and XPs.

    The shotlist is auto-loaded based on date or XP, and the shotlist
    can not be modified.

    Machine.filter_shots() returns an ImmutableMachine object.

    **Usage**::

        >>> xp1013 = fdp.nstxu.filter_shots(xp=1013)
        >>> for shot in xp1013:
        ...     shot.mpts.te.plot()
        ...

    """

    def __init__(self, xp=[], date=[], parent=None):
        self._shots = {}
        self._parent = parent
        shotlist = self._parent.get_shotlist(xp=xp, date=date)
        for shot in shotlist:
            self._shots[shot] = getattr(self._parent, 's{}'.format(shot))

    def __getattr__(self, name):
        try:
            shot = int(name.split('s')[1])
            return self._shots[shot]
        except:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                type(self), name))

    def __repr__(self):
        return '<immutable machine {}>'.format(self._name.upper())

    def __iter__(self):
        return iter(self._shots.values())

    def __contains__(self, value):
        return value in self._shots

    def __len__(self):
        return len(self._shots.keys())

    def __getitem__(self, item):
        pass

    def __dir__(self):
        return ['s{}'.format(shot) for shot in self._shots]

    def logbook(self):
        for shotnum in self._shots:
            shotObj = self._shots[shotnum]
            shotObj.logbook()

    def list_shots(self):
        for shotnum in self._shots:
            shotObj = self._shots[shotnum]
            print('{} in XP {} on {}'.format(
                shotObj.shot, shotObj.xp, shotObj.date))
