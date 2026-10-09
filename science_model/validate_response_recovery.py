"""Independent vacuum-antenna energy and receive-normalization checks.

Uses the isolated PEC 6.001 m reference solved at 5 and 25 MHz. --solve
recreates that reference from the proposal's existing deck generator.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None
from recover_response import ROOT, C, ETA, KB, parse_fields, antenna_report, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solve', action='store_true')
    p.add_argument('--cell-m', type=float, default=0.05)
    p.add_argument('--reference-name', default='vacuum_reference')
    p.add_argument('--disable-port-solver', action='store_true')
    p.add_argument('--frequencies-mhz', type=float, nargs='+', default=[5, 25])
    p.add_argument('--super-low-frequency', action='store_true',
                   help='Experimental engine setting MOMSDB_USE_SUPERLOWFREQ; retain separate reference outputs')
    p.add_argument('--config', action='append', default=[], help='Experimental KEY=VALUE in local momentum.cfg')
    args = p.parse_args()
    ads = Path('C:/Program Files/Keysight/ADS2026_Update1')
    engine = ads/'Momentum/2026.10/win32_64/bin/MomEngine.exe'
    env = os.environ.copy(); env['HPEESOF_DIR'] = str(ads)
    env['PATH'] = str(ads/'bin')+os.pathsep+env['PATH']
    if '/' in args.reference_name or '\\' in args.reference_name or args.reference_name in ('.', '..'):
        raise ValueError('Reference name must be one directory name')
    root = ROOT/'results/response_recovery'/args.reference_name
    work = root/'cases/er01p0_h00000mm'
    if args.solve:
        proposal = ROOT.parents[2]/'external/em_archive/ads_momentum_benchmark'
        subprocess.run([sys.executable, str(proposal/'prepare_stage_a_case.py'),
                        str(proposal/'output/simulation_input'), str(root), '--height-m', '0',
                        '--er', '1', '--frequencies-mhz', *map(str, args.frequencies_mhz),
                        '--max-cell-m', str(args.cell_m)], check=True)
        (root/'data').mkdir(exist_ok=True)
        if args.super_low_frequency:
            with (work/'momentum.cfg').open('a') as cfg:
                cfg.write('\nMOMSDB_USE_SUPERLOWFREQ=TRUE\n')
        if args.config:
            with (work/'momentum.cfg').open('a') as cfg:
                cfg.write('\n'+'\n'.join(args.config)+'\n')
        if args.disable_port_solver:
            opt = work/'proj.opt'
            opt.write_text(opt.read_text().replace('INCLUDEPORTSOLVER ON;', 'INCLUDEPORTSOLVER OFF;'))
        subprocess.run([str(engine), '-O', '-3D', '--objMode=MW', '--dsName=Vacuum_reference',
                        '--dsDir='+str(root/'data'), 'proj', 'proj'], cwd=work, env=env, check=True)
    dsfile = root/'data/Vacuum_reference.ds'
    with dataset.open_dataset_for_reading(dsfile) as ds:
        frame = ds['data'].to_dataframe()
    checks = []
    for f in frame.index:
        s = complex(frame.at[f, 'S[1,1]'])
        z = 50*(1+s)/(1-s)
        current = 1/(z+50)
        plan = ('VISUALIZATIONTYPE 1;\nPARAMETER FREQUENCY, UNITS GHz,\n'
                f'PT {f/1e9};\nPARAMETER PHI, UNITS DEG, PT 0;\nVAR THETA, UNITS DEG, PT 0;\n'
                'PORT 1,\nUNITS VOLT, UNITS DEG, AMPLITUDE 1 PHASE 0,\n'
                'UNITS OHM, UNITS RAD, AMPLITUDE 50 PHASE 0;\n')
        (work/'proj.vpl').write_text(plan)
        subprocess.run([str(engine), '-FF', 'proj', 'proj'], cwd=work, env=env, check=True,
                        capture_output=True)
        report = antenna_report(work/'proj.ant')
        angle, field = parse_fields(work/'proj.fff')
        field *= report['emax_v']
        w = np.sin(np.deg2rad(angle[:,0]))*np.deg2rad(1)**2
        pint = float(np.sum(abs(field)**2*w[:,None])/(2*ETA))
        paccept = 0.5*abs(current)**2*z.real
        # Reciprocal effective length predicts the conventional matched Ae=lambda^2 D/(4pi).
        h = field/current*4*np.pi/(1j*(2*np.pi*f/C)*ETA)
        ae_h = ETA*np.max(np.sum(abs(h)**2,axis=1))/(4*z.real)
        ae_d = (C/f)**2*10**(report['directivity_dbi']/10)/(4*np.pi)
        # Isotropic blackbody voltage PSD must recover the one-sided Nyquist result.
        # One-sided RMS electric-field PSD per polarization is kT*eta/lambda^2.
        vsky = KB*300*ETA/(C/f)**2*np.sum(abs(h)**2*w[:,None])
        row = dict(frequency_mhz=float(f/1e6), impedance_real_ohm=z.real,
                   integrated_power_over_accepted=pint/paccept,
                   report_power_over_accepted=report['radiated_power_w']/paccept,
                   effective_area_reciprocity_over_directivity=ae_h/ae_d,
                   isotropic_noise_over_nyquist=float(vsky/(4*KB*300*z.real)))
        checks.append(row)
    # No renormalization is used to force these checks to pass.
    passed = all(abs(r[k]-1)<0.02 for r in checks for k in (
        'integrated_power_over_accepted', 'effective_area_reciprocity_over_directivity',
        'isotropic_noise_over_nyquist'))
    payload = {'checks': checks, 'passed_2percent': passed,
               'maximum_cell_m': json.loads((work/'case.json').read_text())['maximum_cell_m'],
               'port_solver_enabled': 'INCLUDEPORTSOLVER ON;' in (work/'proj.opt').read_text(),
               'momentum_config': (work/'momentum.cfg').read_text(),
               'substrate_query': (work/'proj.qry').read_text(),
               'dataset_sha256': digest(dsfile),
               'limitations': 'Transmit/receive reciprocity theorem and area/noise consistency, not an independent receive-mode solve; no mesh-convergence claim.'}
    (root/'validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
