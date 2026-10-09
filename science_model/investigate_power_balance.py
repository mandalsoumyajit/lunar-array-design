"""Controlled source-load experiment to locate field/network disagreement.

In a linear one-port antenna, E(Rs)=G*Vs/(Z+Rs). Fitting 1/E against Rs
therefore infers the impedance used by far-field excitation, independently
of a dataset S-to-Z conversion. All exports run in separate local copies.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT, ETA, parse_fields, antenna_report, write_csv, digest

ADS = Path('C:/Program Files/Keysight/ADS2026_Update1')
ENGINE = ADS/'Momentum/2026.10/win32_64/bin/MomEngine.exe'


def environment():
    env = os.environ.copy()
    env['HPEESOF_DIR'] = str(ADS)
    env['PATH'] = str(ADS/'bin')+os.pathsep+env['PATH']
    return env


def export(work, out, f, rs, vs=1.):
    name = f'f{f/1e6:g}_r{rs:g}_v{vs:g}'
    plan = ('VISUALIZATIONTYPE 1;\nPARAMETER FREQUENCY, UNITS GHz,\n'
            f'PT {f/1e9:.17g};\nPARAMETER PHI, UNITS DEG, PT 0;\n'
            'VAR THETA, UNITS DEG, PT 0;\nPORT 1,\n'
            f'UNITS VOLT, UNITS DEG, AMPLITUDE {vs:.17g} PHASE 0,\n'
            f'UNITS OHM, UNITS RAD, AMPLITUDE {rs:.17g} PHASE 0;\n')
    (work/'proj.vpl').write_text(plan)
    result = subprocess.run([str(ENGINE),'-FF','proj','proj'],cwd=work,
                            env=environment(),capture_output=True,text=True,check=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/(name+'.log')).write_text(result.stdout+result.stderr)
    for suffix in ('vpl','ant','fff'):
        shutil.copy2(work/('proj.'+suffix),out/(name+'.'+suffix))
    text=(work/'proj.ant').read_text()
    assert f'Frequency {f/1e9:g} GHz' in text
    angles, fields = parse_fields(work/'proj.fff')
    report = antenna_report(work/'proj.ant')
    fields *= report['emax_v']
    idx = np.flatnonzero((angles[:,0]==90)&(angles[:,1]==90))[0]
    weights = np.sin(np.deg2rad(angles[:,0]))*np.deg2rad(1)**2
    power = float(np.sum(abs(fields)**2*weights[:,None])/(2*ETA))
    return fields[idx,1],power,report


def main():
    source = ROOT/'results/response_recovery/vacuum_reference'
    out = ROOT/'results/power_investigation/source_sweep'
    work = out/'work'
    work.mkdir(parents=True,exist_ok=True)
    case = source/'cases/er01p0_h00000mm'
    for file in case.iterdir():
        if file.is_file(): shutil.copy2(file,work/file.name)
    dsfile = source/'data/Vacuum_reference.ds'
    with dataset.open_dataset_for_reading(dsfile) as ds:
        frame = ds['data'].to_dataframe()
    rows, summaries = [], []
    for f in frame.index:
        s = complex(frame.at[f,'S[1,1]']); z=50*(1+s)/(1-s)
        resistances=np.array([1.,50.,500.,5000.])
        evalues=[]
        for rs in resistances:
            e,power,report=export(work,out,f,rs)
            evalues.append(e)
            current=1/(z+rs)
            rows.append(dict(frequency_mhz=f/1e6,source_resistance_ohm=rs,
                        z_dataset_real=z.real,z_dataset_imag=z.imag,
                        e_real_v=e.real,e_imag_v=e.imag,
                        integrated_power_w=power,report_power_w=report['radiated_power_w'],
                        accepted_power_w=abs(current)**2*z.real/2,
                        radiation_resistance_inferred_ohm=2*power/abs(current)**2))
        y=1/np.array(evalues)
        X=np.column_stack([np.ones(4),resistances])
        coeff=np.linalg.lstsq(X,y,rcond=None)[0]
        zff=coeff[0]/coeff[1]
        residual=np.linalg.norm(X@coeff-y)/np.linalg.norm(y)
        ev2,_,_=export(work,out,f,50,2.)
        summaries.append(dict(frequency_mhz=f/1e6,z_dataset_real=z.real,z_dataset_imag=z.imag,
                         z_from_field_load_sweep_real=zff.real,z_from_field_load_sweep_imag=zff.imag,
                         fractional_impedance_difference=abs(zff-z)/abs(z),
                         reciprocal_field_linear_fit_relative_error=float(residual),
                         double_voltage_field_ratio_real=(ev2/evalues[1]).real,
                         double_voltage_field_ratio_imag=(ev2/evalues[1]).imag))
    write_csv(out/'source_sweep.csv',rows)
    payload=dict(dataset_sha256=digest(dsfile),source_sweep=summaries)
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
