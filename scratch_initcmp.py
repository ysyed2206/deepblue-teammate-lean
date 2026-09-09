"""Cold start for two builds, measured back to back on the same machine state.

Absolute numbers have proven unreliable -- the same build read 62.6s this
morning and 99.6s this evening with no other job running, most likely sustained
thermal throttling after hours of matches. What survives that is the RATIO
between two builds measured minutes apart under identical conditions.
"""
import os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(r"C:\Users\uniqu\Downloads\deepblue-teammate-lean")
agent = ROOT / "agent.py"
backup = agent.read_text(encoding="utf-8")
try:
    for target in ("fastsearch131", "fastsearch140"):
        n = target.replace("fastsearch", "")
        text = re.sub(r"fastsearch1\d\d", target, backup)
        text = re.sub(r"FastEngine1\d\d", "FastEngine" + n, text)
        agent.write_text(text, encoding="utf-8")
        cache = rf"C:/Users/uniqu/AppData/Local/Temp/cs_{n}"
        shutil.rmtree(cache, ignore_errors=True)
        env = {**os.environ, "NUMBA_CACHE_DIR": cache}
        t = time.time()
        r = subprocess.run([sys.executable, "-c", "import agent; print('ok')"],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        print("%-14s cold start %6.1fs   %s"
              % (target, time.time() - t, r.stdout.strip() or r.stderr.strip()[:70]))
finally:
    agent.write_text(backup, encoding="utf-8")
    print("agent.py restored")
