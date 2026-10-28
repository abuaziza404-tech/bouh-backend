"""One durable worker per persistent volume; child timeout isolates GEE hangs."""
import fcntl
import logging
import multiprocessing
import os
import threading
import time

LOG=logging.getLogger('bouh.worker')


def child(payload,path):
    if payload.get('_processor')=='coverage':
        from .coverage_routes import process_coverage
        process_coverage(payload,path)
    else:
        from .gee_adapter import run_analysis
        run_analysis(payload,path)


class Worker:
    def __init__(self,store):
        self.store=store; self.stop_event=threading.Event(); self.thread=None
        self.lock=open(store.directory/'worker.lock','a+')
        # Fail explicitly on a second server sharing this DB. Do not recover
        # another live worker's jobs. Intended deployment: one uvicorn process.
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def start(self):
        self.store.recover()
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def loop(self):
        import json
        ctx=multiprocessing.get_context('spawn')
        timeout=int(os.getenv('BOUH_JOB_TIMEOUT_SECONDS','1200'))
        while not self.stop_event.is_set():
            job=self.store.claim()
            if not job:
                self.stop_event.wait(2); continue
            ident=job['id']; path=self.store.result_path(ident)
            proc=ctx.Process(target=child,args=(json.loads(job['payload']),str(path)))
            try:
                proc.start(); deadline=time.monotonic()+timeout
                while proc.is_alive() and time.monotonic()<deadline and not self.stop_event.is_set():
                    proc.join(timeout=1)
                if proc.is_alive():
                    proc.terminate(); proc.join(timeout=5)
                    if proc.is_alive(): proc.kill(); proc.join(timeout=2)
                    self.store.finish(ident,'failed','PROCESSING_TIMEOUT_OR_SHUTDOWN')
                elif proc.exitcode==0 and path.exists():
                    self.store.finish(ident,'completed')
                else:
                    self.store.finish(ident,'failed','PROCESSING_OR_QUALITY_GATE_FAILED')
            except Exception:
                LOG.exception('Job %s failed',ident)
                if proc.is_alive(): proc.terminate(); proc.join(timeout=5)
                self.store.finish(ident,'failed','WORKER_ERROR')

    def close(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=10)
        self.lock.close()
