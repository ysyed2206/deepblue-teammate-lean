"""Hold the system awake while long chess matches run.

The active power scheme already has STANDBYIDLE at 0 on both AC and DC, so the
sleep timeout is not what is putting this machine out -- Modern Standby (S0ix)
suspends regardless of that setting. SetThreadExecutionState is the documented
way to say "keep running", and it lapses the moment this process exits, so
nothing needs undoing afterwards.

ES_CONTINUOUS | ES_SYSTEM_REQUIRED keeps the SYSTEM awake but lets the DISPLAY
sleep, so the screen still turns off normally.
"""
import ctypes
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print("keep-awake active (system stays up, display may still sleep)", flush=True)
while True:
    # Re-assert periodically; some power transitions clear the flag.
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    time.sleep(60)
