import subprocess, sys
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
ok = 0
for lib, fn in ((ctypes.windll.kernel32, 'K32GetProcessMemoryInfo'),
                (ctypes.windll.psapi, 'GetProcessMemoryInfo')):
    try:
        ok = getattr(lib, fn)(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        if ok: break
    except Exception: pass
print('PEAK_MB %.0f OK=%d' % (c.PeakWorkingSetSize / 1048576, ok))
"""
for mod in ("fastsearch131", "fastsearch134"):
    n = mod.replace("fastsearch", "")
    r = subprocess.run([sys.executable, "-c", code.format(mod=mod, n=n)],
                       cwd=ROOT, capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if l.startswith("PEAK_MB")]
    print("%-14s %s" % (mod, out[0] if out else "ERR " + (r.stderr.strip()[-120:] or "no output")))
