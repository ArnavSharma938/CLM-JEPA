"""Wait for the kernel's process-exit event; no status polling or sleep loop."""
import json
import os
from pathlib import Path
import select

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/decoder_projected/confirmation_second640'
pid = json.loads((OUT/'process.json').read_text())['pid']
try:
    fd = os.pidfd_open(pid)
except ProcessLookupError:
    fd = None
if fd is not None:
    try:
        select.select([fd],[],[])
    finally:
        os.close(fd)
verified = OUT/'all_evaluations_verified.json'
print(json.dumps({'event':'confirmation_process_exited','pid':pid,'all_evaluations_verified':verified.exists()}),flush=True)
if not verified.exists():
    print('\n'.join((OUT/'orchestrator.log').read_text().splitlines()[-16:]),flush=True)
    raise SystemExit(1)
