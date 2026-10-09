"""Add receiver-study edge/intermediate samples without changing prior outputs."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from physical_station_solver import ROOT,prepare_vacuum
from validate_physical_station import verify_case
from export_physical_response import export

frequencies=[14,35,45,46]
for f in frequencies:prepare_vacuum(f*1e6)

def run_frequency(f):
    for material in ['layered','lossy']:
        work=ROOT/f'physical_{material}_f{f}'
        try:verify_case(work)
        except (OSError,RuntimeError,KeyError):
            command=[sys.executable,str(Path(__file__).with_name('physical_station_solver.py')),
                     '--'+material,'--frequency-mhz',str(f)]
            result=subprocess.run(command,capture_output=True,text=True)
            work.mkdir(exist_ok=True);(work/'run.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(f'Failed {work}; see run.log')
        record=verify_case(work)
        print(f,material,'power error',record['worst_power_error'],flush=True)
    return f

with ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(run_frequency,frequencies))
for f in completed:export(ROOT/f'physical_lossy_f{f}')
print('Receiver grid extension complete',completed,flush=True)
