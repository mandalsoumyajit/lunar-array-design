"""Rebuild and audit the physical station power-closure suite."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from physical_station_solver import ROOT,prepare_vacuum
from recover_response import digest


def verify_case(directory):
    """Check a station case. Native Momentum inputs are re-verified when present.

    The public package omits the raw Momentum projects. Setting the environment
    variable LUNAR_ARRAY_PROCESSED_DATA=1 accepts the shipped, hash-checked
    solution without re-verifying those absent native inputs.
    """
    import os
    processed_only=os.environ.get('LUNAR_ARRAY_PROCESSED_DATA')=='1'
    record=json.loads((directory/'validation.json').read_text())
    for path,expected in record['provenance'].items():
        if not Path(path).exists():
            if processed_only:continue
            raise RuntimeError(f'Native solver input not found: {path}. Set LUNAR_ARRAY_PROCESSED_DATA=1 '
                               'to use the shipped processed responses without re-verifying the native Momentum solves.')
        if digest(Path(path))!=expected:raise RuntimeError(f'Stale physical operator input: {path}')
        if Path(path).name=='solve_manifest.json':
            from momentum_safe import verify_solve
            verify_solve(Path(path).parent,[record['frequency_hz']])
    if digest(directory/'solution.npz')!=record['solution_sha256']:raise RuntimeError('Changed physical solution')
    if not record['passed_power_closure']:raise RuntimeError(f'Power closure failed: {directory.name}')
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    frequencies=[1,5,10,15,20,25,30,40,50]
    jobs=[(f,lossy,[]) for f in frequencies for lossy in [False,True]]
    jobs += [(25,False,['--order','48','--nphi','144','--field-order','64','--tag','fine']),
             (50,False,['--order','48','--nphi','144','--field-order','64','--tag','fine']),
             (1,True,['--order','64','--nphi','144','--field-order','64','--tag','fine']),
             (25,False,['--mesh-scale','2','--tag','mesh2'])]
    if args.run:
        # License use is serial; subsequent operator/field jobs use no license.
        for f in frequencies:prepare_vacuum(f*1e6)
        prepare_vacuum(25e6,2)
    def job(spec):
        f,lossy,extra=spec
        label=f'physical_{"lossy" if lossy else "layered"}_f{f}'
        if '--tag' in extra:label+='_'+extra[extra.index('--tag')+1]
        directory=ROOT/label
        if args.run:
            try:verify_case(directory)
            except (OSError,KeyError,RuntimeError):
                command=[sys.executable,str(Path(__file__).with_name('physical_station_solver.py')),
                         '--lossy' if lossy else '--layered','--frequency-mhz',str(f),*extra]
                result=subprocess.run(command,capture_output=True,text=True)
                directory.mkdir(exist_ok=True)
                (directory/'run.log').write_text(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError(f'Failed {label}; see run.log')
        r=verify_case(directory)
        print(label,'worst power error',r['worst_power_error'],flush=True)
        return label,r
    # Keep cases sharing a vacuum cache in one worker, avoiding concurrent writes.
    groups=[[spec for spec in jobs if spec[0]==f] for f in frequencies]
    def group(specs):return [job(spec) for spec in specs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=dict(item for results in pool.map(group,groups) for item in results)
    comparisons={}
    for f,tag in [(25,'fine'),(50,'fine'),(25,'mesh2')]:
        with np.load(ROOT/f'physical_layered_f{f}'/'solution.npz') as d:z0=d['z']
        with np.load(ROOT/f'physical_layered_f{f}_{tag}'/'solution.npz') as d:z1=d['z']
        comparisons[f'f{f}_{tag}']=float(np.linalg.norm(z1-z0)/np.linalg.norm(z1))
    assert comparisons['f25_fine']<.001 and comparisons['f50_fine']<.001,comparisons
    assert comparisons['f25_mesh2']<.01,comparisons
    with np.load(ROOT/'physical_lossy_f1/solution.npz') as d:z0=d['z']
    with np.load(ROOT/'physical_lossy_f1_fine/solution.npz') as d:z1=d['z']
    comparisons['f1_lossy_fine']=float(np.linalg.norm(z1-z0)/np.linalg.norm(z1))
    assert comparisons['f1_lossy_fine']<.001,comparisons
    audit=dict(passed=True,frequencies_mhz=frequencies,case_count=len(records),
               maximum_power_error=max(r['worst_power_error'] for r in records.values()),
               impedance_convergence_relative=comparisons,
               cases={name:{key:r[key] for key in ['frequency_hz','lossy','power_eigenvalues','worst_power_error','lower_farfield_vs_surface_flux_relative_error','reciprocity_relative_error']} for name,r in records.items()},
               validation_sha256={name:digest(ROOT/name/'validation.json') for name in records},
               scope='Numerical power closure for the stated ideal PEC/feed/planar-ground model; not measured instrument performance.')
    (ROOT/'physical_station_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({key:value for key,value in audit.items() if key not in ['cases','validation_sha256']},indent=2))

if __name__=='__main__':main()
