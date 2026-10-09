"""Run Momentum with an explicit local database and auditable frequency coverage."""
from pathlib import Path
import json
import re
import subprocess
from investigate_power_balance import ENGINE, environment
from recover_response import digest

INPUTS=('proj_a','proj.pin','proj.prt','proj.ltd','proj.opt','proj.sti','proj.cfg','momentum.cfg')


def configure(work, frequencies_hz, export_currents=False):
    work=Path(work)
    (work/'local_cache').mkdir(exist_ok=True)
    # Relative paths avoid the legacy configuration parser's handling of spaces.
    # All three search locations are local: no fallback to a shared database.
    (work/'proj.cfg').write_text('user = ./local_cache/\nsite = ./local_cache/\nsupplied = ./local_cache/\n')
    floor=min(frequencies_hz)/1e9/10
    config=f'MOMSDB_USE_QSFREQUENCY={floor:.17g}\n'
    if export_currents:
        config+='MOM3D_WRITE_MATLABFILES=TRUE\nMOM3D_WRITE_BASISSECTIONS=TRUE\n'
    (work/'momentum.cfg').write_text(config)


def database_manifest(work, frequencies_hz):
    work=Path(work).resolve()
    query=(work/'proj.qry').read_text()
    match=re.search(r'^Frequency range : from (\S+) GHz to (\S+) GHz\.',query,re.M)
    if not match:raise RuntimeError('Missing actual substrate-database frequency coverage')
    low,high=(float(v)*1e9 for v in match.groups())
    if low>min(frequencies_hz)*(1+1e-9) or high<max(frequencies_hz)*(1-1e-9):
        raise RuntimeError(f'Substrate database [{low}, {high}] Hz does not cover requested band')
    raw=re.search(r'^Corresponding potentialname : (.+)$',query,re.M)[1].strip()
    db=Path(raw)
    if not db.is_absolute():db=work/db
    db=db.resolve()
    if not db.is_relative_to(work/'local_cache'):
        raise RuntimeError(f'Substrate database escaped the case-local cache: {db}')
    return dict(actual_min_hz=low,actual_max_hz=high,database_path=str(db),
                database_sha256=digest(db),query=query)


def solve(work, frequencies_hz, name='Corrected', export_currents=False):
    work=Path(work).resolve()
    configure(work,frequencies_hz,export_currents)
    # Validate the database before allowing the network solve.
    commands=[['-O','-DB','--objMode=MW','proj','proj'],
              ['-O','-3D','--objMode=MW','--dsName='+name,'--dsDir='+str(work),'proj','proj']]
    for index,command in enumerate(commands):
        result=subprocess.run([str(ENGINE),*command],cwd=work,env=environment(),capture_output=True,text=True)
        log=result.stdout+result.stderr
        (work/f'safe_step_{index}.log').write_text(log)
        if result.returncode or '--- ERROR' in log:
            raise RuntimeError(f'Momentum failed; see {work}/safe_step_{index}.log')
        manifest=database_manifest(work,frequencies_hz)
    manifest.update(engine_sha256=digest(ENGINE),requested_hz=list(frequencies_hz),
                    inputs={p:digest(work/p) for p in INPUTS},dataset_sha256=digest(work/(name+'.ds')))
    if export_currents:
        manifest['current_exports']={p.name:digest(p) for p in
            [work/'proj.msh',work/'BasisSections.log',*work.glob('Imatrix.m.*')]}
    (work/'solve_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return work/(name+'.ds')


def verify_solve(work, frequencies_hz, name='Corrected'):
    """Reject stale cached networks; verify current exports where recorded."""
    work=Path(work)
    manifest=json.loads((work/'solve_manifest.json').read_text())
    actual=database_manifest(work,frequencies_hz)
    if actual['database_sha256']!=manifest['database_sha256']:
        raise RuntimeError('Cached substrate database changed')
    for name_,expected in manifest['inputs'].items():
        if digest(work/name_)!=expected:raise RuntimeError(f'Cached solve input changed: {name_}')
    for name_,expected in manifest.get('current_exports',{}).items():
        if digest(work/name_)!=expected:raise RuntimeError(f'Cached current export changed: {name_}')
    if digest(work/(name+'.ds'))!=manifest['dataset_sha256']:
        raise RuntimeError('Cached dataset changed')
    if digest(ENGINE)!=manifest['engine_sha256']:raise RuntimeError('Solver executable changed')
    return manifest


def parse_cut(path):
    """Read one absolute-field CITI cut (VISUALIZATIONTYPE 0, CLIP -300)."""
    import numpy as np
    text=Path(path).read_text()
    names=re.findall(r'^DATA (\S+) RI',text,re.M)
    blocks=re.findall(r'^BEGIN\s*\n(.*?)^END\s*$',text,re.M|re.S)
    angles=np.array([float(v) for v in re.search(r'VAR_LIST_BEGIN\s*\n(.*?)VAR_LIST_END',text,re.S)[1].split()])
    data={name:np.array([complex(*map(float,line.split(','))) for line in block.splitlines() if line.strip()]) for name,block in zip(names,blocks)}
    if len(data)!=len(names) or any(len(v)!=len(angles) for v in data.values()):
        raise RuntimeError('Unsupported CITI cut structure')
    return angles,np.column_stack([data['Etheta'],data['Ephi']])


def absolute_cut(work, plan, output):
    """Use the documented absolute Etheta/Ephi cut quantities, suppress display clipping."""
    import shutil
    work=Path(work);output=Path(output)
    (work/'proj.vpl').write_text('CLIP -300;\n'+plan.replace('VISUALIZATIONTYPE 1;','VISUALIZATIONTYPE 0;'))
    result=subprocess.run([str(ENGINE),'-FF','proj','proj'],cwd=work,env=environment(),capture_output=True,text=True)
    if result.returncode or '--- ERROR' in result.stdout+result.stderr:raise RuntimeError('Absolute far-field cut failed')
    shutil.copy2(work/'proj.ctf',output)
    shutil.copy2(work/'proj.vpl',output.with_suffix('.vpl'))
    return parse_cut(output)
