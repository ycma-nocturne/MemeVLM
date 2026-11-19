"""Record host process resources during model setup and execution.

Write invocation receipts separately from the attempt ledger and retain
incomplete records for interrupted processes.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _save_new(path,value):
    raw=_bytes(value)
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


class ResourceInvocation:
    def __init__(self, *, runs, metadata, snapshot, read_snapshot):
        root=Path(runs).resolve()
        self.directory=root.parent/(root.name+'.profiles')/uuid.uuid4().hex
        self.read_snapshot=read_snapshot
        self.runs=root
        self.started={'schema_version':1,'protocol_id':'reference_reuse_v1',
            'invocation_id':self.directory.name,'runs_path':str(root),**metadata,
            'attempts_before':snapshot['attempts'],'ledger_before_sha256':snapshot['ledger_sha256'],
            'ledger_bytes_before':snapshot['ledger_size_bytes'],
            'timing_scope':'backend construction, setup, execution and ledger bookkeeping; excludes input preflight and lock wait'}
        self.setup_seconds=0.0
        self.setup_attempted=self.setup_completed=False
        self.resource_after_setup={'available':False,'reason':'backend_setup_not_completed'}
        self.result=None

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=False)
        self.started['started_at']=_now()
        self.started_sha256=_save_new(self.directory/'started.json',self.started)
        self.start=time.perf_counter()
        return self

    @contextmanager
    def setup(self,attempted):
        self.setup_attempted=bool(attempted)
        start=time.perf_counter()
        try:
            yield
            self.setup_completed=bool(attempted)
        finally:
            self.setup_seconds=time.perf_counter()-start if attempted else 0.0

    def resources(self,backend):
        snapshot=getattr(backend,'resource_snapshot',None)
        self.resource_after_setup=(snapshot() if callable(snapshot) else
            {'available':False,'reason':'backend_does_not_expose_resource_snapshot'})

    def __exit__(self,error_type,error,traceback):
        wall_seconds=time.perf_counter()-self.start
        try:
            after=self.read_snapshot(self.runs)
            after_fields={'attempts_after':after['attempts'],'ledger_after_sha256':after['ledger_sha256'],
                          'ledger_bytes_after':after['ledger_size_bytes']}
        except Exception as read_error:
            after_fields={'attempts_after':None,'ledger_after_sha256':None,'ledger_bytes_after':None,
                          'ledger_snapshot_error':{'type':type(read_error).__name__,'message':str(read_error)}}
        finished={'schema_version':1,'protocol_id':'reference_reuse_v1',
            'invocation_id':self.directory.name,'started_sha256':self.started_sha256,
            'status':'error' if error_type else 'completed','finished_at':_now(),
            'wall_seconds':wall_seconds,'setup_seconds':self.setup_seconds,
            'setup_attempted':self.setup_attempted,'setup_completed':self.setup_completed,
            'resource_after_setup':self.resource_after_setup,**after_fields,
            'output_summary':self.result,
            'error':{'type':error_type.__name__,'message':str(error)} if error_type else None,
            'profile_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        _save_new(self.directory/'finished.json',finished)
        return False
