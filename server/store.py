"""Durable single-instance SQLite job queue. Transactions enforce admission."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid


class AdmissionError(RuntimeError): pass
class IdempotencyConflict(RuntimeError): pass


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = self.directory/'jobs.sqlite3'
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, idem TEXT NOT NULL,
                payload TEXT NOT NULL, digest TEXT NOT NULL, state TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL, error TEXT,
                UNIQUE(owner,idem))''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db, timeout=10)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, owner, idem, payload):
        text=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
        digest=hashlib.sha256(text.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT * FROM jobs WHERE owner=? AND idem=?',(owner,idem)).fetchone()
            if prior:
                if prior['digest'] != digest: raise IdempotencyConflict()
                return dict(prior)
            active=db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
            own=db.execute("SELECT count(*) FROM jobs WHERE owner=? AND state IN ('queued','running')",(owner,)).fetchone()[0]
            if active>=6 or own>=2: raise AdmissionError('Queue full; retry after an active job finishes')
            ident=uuid.uuid4().hex; now=time.time()
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',
                       (ident,owner,idem,text,digest,'queued',now,now,None))
            return dict(db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone())

    def get(self, ident, owner):
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=? AND owner=?',(ident,owner)).fetchone()
            return dict(row) if row else None

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?",(time.time(),row['id']))
                return dict(row)

    def finish(self, ident, state, error=None):
        if state not in {'completed','failed'}: raise ValueError('Invalid terminal state')
        with self.connect() as db:
            db.execute('UPDATE jobs SET state=?,updated=?,error=? WHERE id=?',
                       (state,time.time(),error,ident))

    def recover(self):
        # Called ONLY while owning the exclusive worker file lock.
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='failed',error='WORKER_RESTARTED',updated=? WHERE state='running'",(time.time(),))

    def result_path(self, ident):
        if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('Invalid job id')
        return self.directory/(ident+'.geojson')
