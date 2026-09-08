"""Start the single resumable confirmation orchestrator and retain its PID."""
import os
import select
from pathlib import Path
import subprocess
import sys
from run_second640_confirmation import ROOT, OUT, verify_inputs, write

verify_inputs()
pidfile = OUT/'process.json'
if pidfile.exists():
    import json
    pid = json.loads(pidfile.read_text())['pid']
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        pass
    else:
        try:
            if not select.select([fd],[],[],0)[0]:
                raise SystemExit(f'Orchestrator PID {pid} is still running; refusing duplicate launch')
        finally:
            os.close(fd)
with (OUT/'orchestrator.log').open('a') as log:
    p = subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_second640_confirmation.py'),'run'],
                         cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(pidfile,{'pid':p.pid,'log':str(OUT/'orchestrator.log')})
print(f'Orchestrator PID {p.pid}',flush=True)
