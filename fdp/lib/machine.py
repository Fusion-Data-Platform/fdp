# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:05:14 2015

@author: ktritz
"""
from __future__ import print_function
from builtins import str, map, range
from collections import Sized, Iterable, Container, deque
import os
from warnings import warn
import numpy as np
import MDSplus as mds
import threading

from .logbook import Logbook
from .parse import parse_top, parse_machine
from .shot import Shot
from .globals import FDP_DIR, FdpError, FdpWarning
from .datasources import canonicalMachineName, MDS_SERVERS, EVENT_SERVERS


def machineClassFactory(name=''):
    """
    Class factory to implement abstract machine class for specific machine
    """
    machine_name = canonicalMachineName(name)
    class_name = 'Machine' + machine_name.capitalize()
    MachineClass = type(class_name, (Machine,), {})
    MachineClass._name = machine_name
    parse_top(MachineClass)
    parse_machine(MachineClass)
    return MachineClass


class Machine(Sized, Iterable, Container):
    """
    Abstract base class for top-level machines
    """

    _connections = None
    _connected = False
    _thread_events = None
    _logbook = None
    _modules = None
    _parent = None
    _name = None

    def __init__(self, shotlist=None, xp=None, date=None):
        self._shots = {}  # shot dictionary with shot number (int) keys
        self._make_server_connections()
        self._set_modules()
        if self._logbook is None:
            self._logbook = Logbook(name=self._name, root=self)
        if shotlist is not None or xp is not None or date is not None:
            self.addshot(shotlist=shotlist, xp=xp, date=date)

    def _make_server_connections(self):
        if self._connections is None:
            mds_server = MDS_SERVERS[self._name]
            hostname = mds_server['hostname']
            port = mds_server['port']
            nconnections = 2
            self._connections = []
            self._thread_events = []
            for i in range(nconnections):
                self._connections.append(None)
                self._thread_events.append(threading.Event())
            def connection_wrapper(i):
                connection = mds.Connection('{}:{}'.format(hostname, port))
                connection.tree = None
                self._connections[i] = connection
                ev = self._thread_events[i]
                ev.set()
            for i in range(nconnections):
                t = threading.Thread(target=connection_wrapper, args=(i,))
                t.start()
        # event_server = EVENT_SERVERS[self._name]
        # self._eventConnection = mds.Connection('{}:{}'.format(event_server['hostname'],
        #                                                       event_server['port']))

    def _set_modules(self):
        if self._modules is None:
            machine_diag_dir = os.path.join(FDP_DIR, 'diagnostics', self._name)
            self._modules = []
            for module in os.listdir(machine_diag_dir):
                diag_dir = os.path.join(machine_diag_dir, module)
                if os.path.isdir(diag_dir) and module[0] is not '_':
                    self._modules.append(module)

    def _validate_shot(self, shot):
        if shot not in self._shots:
            self._shots[shot] = Shot(shot, self)

    def __getattr__(self, attr_name):
        try:
            shot = int(attr_name.split('s')[1])
            self._validate_shot(shot)
            return self._shots[shot]
        except:
            # import pdb
            # pdb.set_trace()
            raise AttributeError('bad attr: {}'.format(attr_name))
            # raise
    
    def __getitem__(self, shot):
        self._validate_shot(shot)
        return self._shots[shot]

    def __delitem__(self, key):
        del self._shots[key]

    def __setitem__(self, item, value):
        pass

    def __dir__(self):
        return ['s{}'.format(shot) for shot in self._shots.keys()]

    def __contains__(self, key):
        return key in self._shots

    def __len__(self):
        return len(self._shots)

    def __iter__(self):
        # iterate over Shot objects in _shots.values()
        # (not over shot numbers in _shots.keys())
        return iter(self._shots.values())

    def __repr__(self):
        return '<machine {}>'.format(self._name.upper())

    def __str__(self):
        return 'Machine {}'.format(self._name.upper())

    def _get_logbook_credentials(self):
        # override with methods/<machine>/_get_logbook_credentials.py
        pass

    def _get_connection(self, shot, tree):
        if not self._connected:
            for ev in self._thread_events:
                ev.wait()
            self._connected = True
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
            # raise
            connection.tree = None
        # finally:
        self._connections.insert(0, connection)
        return connection

    def _get_mdsshape(self, signal):
        # if signal.shot is 0:
        #     print('No MDS data exists for model tree')
        #     return
        connection = self._get_connection(signal.shot, signal.mdstree)
        try:
            usage_code = connection.get('getnci({},"USAGE")'.format(signal.mdsnode)).data()
            length = connection.get('getnci({},"LENGTH")'.format(signal.mdsnode)).data()
            if usage_code != 6 or length < 1:
                raise ValueError
            return connection.get('shape({})'.format(signal.mdsnode)).data()
        except:
            return

    def _get_mdsdata(self, signal):
        shot = signal.shot
        # if shot is 0:
        #     print('No MDS data exists for model tree')
        #     return np.zeros(0)
        connection = self._get_connection(shot, signal.mdstree)
        if signal.mdstree.lower() == 'ptdata':
            if 'Signal' in str(type(signal)):
                mds_address = 'ptdata("{}", {:d})'.format(signal.mdsnode, shot)
            elif 'Axis' in str(type(signal)):
                mds_address = 'dim_of(ptdata("{}", {:d}))'.format(signal.mdsnode, shot)
            else:
                raise FdpError('bad mds data')
        else:
            mds_address = signal.mdsnode
        try:
            data = connection.get(mds_address)
        except:
            # raise
            msg = 'MDSplus connection error for shot {}, tree {}, and node {}'.format(
                signal.shot, signal.mdstree, mds_address)
            raise FdpError(msg)
            # warn(msg, FdpWarning)
            # return np.zeros(0)
        if getattr(signal, '_raw_of', None) is not None:
            data = data.raw_of()
        if getattr(signal, '_dim_of', None) is not None:
            data = data.dim_of()
        data = data.value_of().value
        if signal._transpose is not None:
            data = data.transpose(signal._transpose)
        if hasattr(signal, '_postprocess'):
            data = signal._postprocess(data)
        return data

    def _get_shots(self, xp=None, date=None):
        shots = []
        if date:
            if not isinstance(date, (list, tuple)):
                date = [date]
            shots.extend(self._logbook.get_shotlist(date=list(date)))
        if xp:
            if not isinstance(xp, (list, tuple)):
                xp = [xp]
            shots.extend(self._logbook.get_shotlist(xp=list(xp)))
        return shots

    def addshot(self, shotlist=None, date=None, xp=None):
        """
        Load shots
        """
        shots = []
        if shotlist:
            if not isinstance(shotlist, (list, tuple)):
                shotlist = [shotlist]
            shots.extend(list(shotlist))
        shots.extend(self._get_shots(xp=xp, date=date))
        for shot in shots:
            self._validate_shot(shot)

    def shotlist(self, xp=None, date=None, quiet=False):
        """
        Generate shotlist
        """
        if xp or date:
            shotlist = self._get_shots(xp=xp, date=date)
        else:
            shotlist = list(self._shots.keys())
        shotlist.sort()
        if not quiet:
            for shotnum in shotlist:
                shot = self[shotnum]
                print('{} in XP {} on {}'.format(shot.shot, shot.xp, shot.date))
        return shotlist

    def filter(self, date=None, xp=None):
        """
        Get a AbstractMachine-like object with an immutable shotlist for XP(s)
        or date(s)
        """
        self.addshot(xp=xp, date=date)
        return ImmutableMachine(xp=xp, date=date, parent=self)

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
                    container._set_dynamic_containers()
                    container_queue.extend(list(container._containers.values()))
                    if obj is None or obj.lower() == 'signal':
                        for signal in list(container._signals.values()):
                            if signal._contains(tag):
                                branch_str = '.'.join([signal._get_branch(),
                                                       signal._name])
                                find_list.add(branch_str)
                    if obj is None or obj.lower() == 'axis':
                        for signal in list(container._signals.values()):
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


# machine classes
Nstxu = machineClassFactory('nstxu')
D3D = machineClassFactory('d3d')


class ImmutableMachine(Sized, Iterable, Container):
    """
    An immutable AbstractMachine-like class for dates and XPs.

    The shotlist is auto-loaded based on date or XP, and the shotlist
    can not be modified.

    AbstractMachine.filter_shots() returns an ImmutableMachine object.

    **Usage**::

        >>> xp1013 = fdp.nstxu.filter_shots(xp=1013)
        >>> for shot in xp1013:
        ...     shot.mpts.te.plot()
        ...

    """

    def __init__(self, xp=None, date=None, parent=None):
        self._shots = {}
        self._parent = parent
        self._name = self._parent._name
        shotlist = self._parent.shotlist(xp=xp, date=date, quiet=True)
        for shot in shotlist:
            self._shots[shot] = self._parent[shot]

    def __getattr__(self, name):
        try:
            shot = int(name.split('s')[1])
            return self[shot]
        except:
            raise AttributeError('bad attr: {}'.format(name))

    def __repr__(self):
        return '<immutable machine {}>'.format(self._name.upper())

    def __iter__(self):
        return iter(self._shots.values())

    def __contains__(self, key):
        return key in self._shots

    def __len__(self):
        return len(self._shots)

    def __delitem__(self, item):
        pass

    def __getitem__(self, item):
        return self._shots[item]

    def __dir__(self):
        return ['s{}'.format(shot) for shot in self]

    def shotlist(self, quiet=False):
        shotlist = list(self._shots.keys())
        shotlist.sort()
        if not quiet:
            for shotnum in shotlist:
                shot = self[shotnum]
                print('{} in XP {} on {}'.format(shot.shot, shot.xp, shot.date))
        return shotlist
