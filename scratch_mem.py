"""Peak memory for the shipped engine. The competition cap is 2048 MB and
exceeding it is another instant-loss failure mode like the init budget --
worth checking before adopting a 304 MB hash table."""
import os, subprocess, sys
ROOT = r"C:\Users\uniqu\Downloads\deepblue-teammate-lean"
code = """
import ctypes, ctypes.wintypes as wt
from deepblue.fastcore import from_fen
from deepblue.{mod} import FastEngine{n}
e = FastEngine{n}()
e.search(from_fen('r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 11'), 3000, 4200)
class PMC(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
c = PMC(); c.cb = ctypes.sizeof(c)
ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
print('PEAK_MB %.0f' % (c.PeakWorkingSetSize / 1024 / 1024))
"""
for mod in ("fastsearch131", "fastsearch134"):
    n = mod.replace("fastsearch", "")
    r = subprocess.run([sys.executable, "-c", code.format(mod=mod, n=n)],
                       cwd=ROOT, capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("PEAK_MB")]
    print("%-14s %s MB   (limit 2048)" % (mod, line[0].split()[1] if line else "ERR " + r.stderr.strip()[:60]))
