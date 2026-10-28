"""Durable exact-roster authorization for an ideal local aggregation backend."""

import json
import sqlite3

import numpy as np

from .core import refine


class RosterLedger:
    """Reserve history before release; unresolved reservations remain irreversible."""

    def __init__(self, path, classes, k, capacity, threshold):
        members = [client for group in classes for client in group]
        if (k < 1 or not classes or len(members) != len(set(members))
                or any(len(group) < k for group in classes)):
            raise ValueError('Initial enrolled partition must satisfy k-safety')
        self.k, self.capacity, self.threshold = k, capacity, threshold
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, classes TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, roster TEXT, status TEXT, answer TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, value TEXT)')
        settings = json.dumps([k, capacity, threshold, sorted(members)])
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO settings VALUES (1, ?)', (settings,))
            self.db.execute('INSERT OR IGNORE INTO state VALUES (1, ?)', (json.dumps(classes),))
        if self.db.execute('SELECT value FROM settings WHERE id=1').fetchone()[0] != settings:
            raise ValueError('An existing history ledger cannot change its enrolled population or policy')

    def reserve(self, session, roster, uploaded):
        selected = set(roster)
        if (len(selected) != len(roster) or not selected <= set(uploaded)
                or not self.threshold <= len(selected) <= self.capacity):
            raise ValueError('Roster violates upload, threshold or capacity requirements')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM sessions WHERE id=?', (session,)).fetchone():
                raise ValueError('Session already reserved; recover it instead of authorizing again')
            classes = [set(c) for c in json.loads(self.db.execute('SELECT classes FROM state WHERE id=1').fetchone()[0])]
            if not selected <= set().union(*classes):
                raise ValueError('Roster contains an unenrolled client')
            following = refine(classes, selected)
            if any(len(group) < self.k for group in following):
                raise ValueError('Roster violates two-sided historical safety')
            self.db.execute('INSERT INTO sessions VALUES (?, ?, ?, NULL)',
                            (session, json.dumps(sorted(selected)), 'reserved'))
            self.db.execute('UPDATE state SET classes=? WHERE id=1',
                            (json.dumps([sorted(c) for c in following]),))
            self.db.commit()
        except (ValueError, sqlite3.Error):
            self.db.rollback()
            raise

    def freeze(self, session, included):
        with self.db:
            updated = self.db.execute('UPDATE sessions SET status=? WHERE id=? AND status=? AND roster=?',
                                      ('frozen', session, 'reserved', json.dumps(sorted(included))))
            if updated.rowcount != 1:
                raise ValueError('Exact-roster agreement required before unmasking')

    def release(self, session, updates):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT roster,status,answer FROM sessions WHERE id=?', (session,)).fetchone()
            if row is None:
                raise ValueError('Unreserved session')
            if row[1] == 'released':
                self.db.commit()
                return np.asarray(json.loads(row[2]))
            if row[1] != 'frozen':
                raise ValueError('An unresolved roster cannot release plaintext')
            roster = json.loads(row[0])
            if not set(roster) <= set(updates):
                raise ValueError('Missing authorized client update')
            answer = np.stack([updates[i] for i in roster]).sum(axis=0)
            if not np.isfinite(answer).all():
                raise ValueError('Aggregate contains nonfinite values')
            self.db.execute('UPDATE sessions SET status=?,answer=? WHERE id=?',
                            ('released', json.dumps(answer.tolist()), session))
            self.db.commit()
        except (ValueError, sqlite3.Error):
            self.db.rollback()
            raise
        return answer

    def close(self):
        self.db.close()
