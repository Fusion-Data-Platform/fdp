# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:19:00 2015

@author: ktritz
"""
import datetime
import numpy as np
import pymssql
from .fdp_globals import FdpError
from .datasources import LOGBOOK_CREDENTIALS


class Logbook(object):

    def __init__(self, name='nstxu', root=None):
        self._name = name.lower()
        self._root = root

        self._credentials = {}
        self._table = ''
        self._shotlist_query_prefix = ''
        self._shot_query_prefix = ''

        self._logbook_connection = None
        self._make_logbook_connection()

        # dict of cached logbook entries
        # kw is shot, value is list of logbook entries
        self.logbook = {}

    def _make_logbook_connection(self):
        self._credentials = LOGBOOK_CREDENTIALS[self._name]
        self._table = self._credentials['table']

        self._shotlist_query_prefix = (
            'SELECT DISTINCT rundate, shot, xp, voided '
            'FROM {} WHERE voided IS null').format(self._table)
        self._shot_query_prefix = (
            'SELECT dbkey, username, rundate, shot, xp, topic, text, entered, '
            'voided FROM {} WHERE voided IS null').format(self._table)

        try:
            self._logbook_connection = pymssql.connect(
                server=self._credentials['server'],
                user=self._credentials['username'],
                password=self._credentials['password'],
                database=self._credentials['database'],
                port=self._credentials['port'],
                as_dict=True)
        except:
            print('Attempting logbook server connection as drsmith')
            try:
                self._logbook_connection = pymssql.connect(
                    server=self._credentials['server'],
                    user='drsmith',
                    password=self._credentials['password'],
                    database=self._credentials['database'],
                    port=self._credentials['port'],
                    as_dict=True)
            except:
                txt = '{} logbook connection failed. '.format(
                    self._name.upper())
                txt = txt + 'Server credentials:'
                for key in self._credentials:
                    txt = txt + '  {0}:{1}'.format(key, self._credentials[key])
                raise FdpError(txt)

    def _get_cursor(self):
        try:
            cursor = self._logbook_connection.cursor()
            cursor.execute('SET ROWCOUNT 500')
        except:
            raise FdpError('Cursor error')
        return cursor

    def _shot_query(self, shot=[]):
        cursor = self._get_cursor()
        if shot and not isinstance(shot, list):
            shot = [shot]
        for sh in shot:
            if sh not in self.logbook:
                query = ('{0} and shot={1} '
                         'ORDER BY shot ASC, entered ASC'
                         ).format(self._shot_query_prefix, sh)
                cursor.execute(query)
                rows = cursor.fetchall()  # list of logbook entries
                for row in rows:
                    rundate = repr(row['rundate'])
                    year = rundate[0:4]
                    month = rundate[4:6]
                    day = rundate[6:8]
                    row['rundate'] = datetime.date(int(year), int(month),
                                                   int(day))
                self.logbook[sh] = rows

    def get_shotlist(self, date=None, xp=None, verbose=False):
        # return list of shots for date and/or XP
        cursor = self._get_cursor()
        rows = []
        shotlist = []   # start with empty shotlist
        if not date:
            date = []
        if not xp:
            xp = []
        if date and not isinstance(date, (list,tuple)):      # if it's just a single date
            date = [date]   # put it into a list
        for d in date:
            query = ('{0} and rundate={1} ORDER BY shot ASC'.
                     format(self._shotlist_query_prefix, d))
            cursor.execute(query)
            rows.extend(cursor.fetchall())
        if xp and not isinstance(xp, (list,tuple)):           # if it's just a single xp
            xp = [xp]             # put it into a list
        for x in xp:
            query = ('{0} and xp={1} ORDER BY shot ASC'.
                     format(self._shotlist_query_prefix, x))
            cursor.execute(query)
            rows.extend(cursor.fetchall())
        for row in rows:
            rundate = repr(row['rundate'])
            year = rundate[0:4]
            month = rundate[4:6]
            day = rundate[6:8]
            row['rundate'] = datetime.date(int(year), int(month), int(day))
        # add shots to shotlist
        shotlist.extend([row['shot'] for row in rows if row['shot'] is not None])
        cursor.close()
        return np.unique(shotlist)

    def get_entries(self, shot=None, date=None, xp=None):
        # return list of lobgook entries (dictionaries) for shot(s)
        shotlist = []
        if shot and not isinstance(shot, list):
            shot = [shot]
        if shot:
            shotlist.extend(shot)
        if xp or date:
            shotlist.extend(self.get_shotlist(date=date, xp=xp))
        if shotlist:
            self._shot_query(shot=shotlist)
        entries = []
        for sh in np.unique(shotlist):
            if sh in self.logbook:
                entries.extend(self.logbook[sh])
        return entries
