"""Recover nominal-station networks and audit absolute fields from saved currents.

Run with the ADS Python interpreter. No new mesh or network solve is performed.
Original proposal datasets are read only; -FF runs use local working copies.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import numpy as np
try:  # Keysight ADS Python is needed only to read native Momentum datasets.
    import keysight.ads.dataset as dataset
except ImportError:
    dataset = None

ROOT = Path(__file__).resolve().parent
C = 299792458.0
ETA = 376.730313668
KB = 1.380649e-23
CASES = ('lf6m_hf0p5m_rot000_lfband', 'lf6m_hf0p5m_rot000_highband')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def parse_fields(path):
    a = np.loadtxt(path, comments='#')
    a = a[a[:, 1] < 360]
    return a[:, :2], a[:, 2::2] + 1j*a[:, 3::2]


def antenna_report(path):
    text = path.read_text()
    def value(pattern):
        return float(re.search(pattern, text).group(1))
    et = value(r'E_theta_max ([\deE.+-]+) ;')
    ep = value(r'E_phi_max ([\deE.+-]+)')
    return dict(emax_v=float(np.hypot(et, ep)),
                radiated_power_w=value(r'Power radiated 1 ([\deE.+-]+)'),
                gain_dbi=value(r'Gain ([\deE.+-]+)'),
                directivity_dbi=value(r'Directivity ([\deE.+-]+)'))


def run_fields(work, out, engine, env, f, port, load):
    stem = f'f{f/1e6:06.2f}_p{port+1}'
    ff, ant = out/(stem+'.fff'), out/(stem+'.ant')
    plan = ('VISUALIZATIONTYPE 1;\nPARAMETER FREQUENCY, UNITS GHz,\n'
            f'PT {f/1e9:.17g};\nPARAMETER PHI, UNITS DEG, PT 0;\n'
            'VAR THETA, UNITS DEG, PT 0;\n')
    zs = np.full(4, load, dtype=complex)
    zs[port] = 50
    for p in range(4):
        plan += (f'PORT {p+1},\nUNITS VOLT, UNITS DEG, AMPLITUDE '
                 f'{1 if p == port else 1e-12:.17g} PHASE 0,\n'
                 f'UNITS OHM, UNITS RAD, AMPLITUDE {abs(zs[p]):.17g} '
                 f'PHASE {np.angle(zs[p]):.17g};\n')
    # Cache keyed by exact plan, copied current files and engine hash (manifest).
    planpath = out/(stem+'.vpl')
    if not (ff.exists() and ant.exists() and planpath.exists()
            and planpath.read_text() == plan):
        (work/'proj.vpl').write_text(plan)
        result = subprocess.run([str(engine), '-FF', 'proj', 'proj'],
                                cwd=work, env=env, capture_output=True, text=True)
        (out/(stem+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Far-field export failed: {stem}: {result.stderr}')
        shutil.copy2(work/'proj.fff', ff)
        shutil.copy2(work/'proj.ant', ant)
        planpath.write_text(plan)
    return parse_fields(ff), antenna_report(ant), zs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proposal-root', type=Path, default=ROOT.parents[2]/'external'/'em_archive')
    p.add_argument('--ads-root', type=Path, default=Path('C:/Program Files/Keysight/ADS2026_Update1'))
    args = p.parse_args()
    source = args.proposal_root/'ads_momentum_benchmark/output/crossed_station_pilot'
    output = ROOT/'results/response_recovery'
    output.mkdir(parents=True, exist_ok=True)
    engine = args.ads_root/'Momentum/2026.10/win32_64/bin/MomEngine.exe'
    env = os.environ.copy()
    env['HPEESOF_DIR'] = str(args.ads_root)
    env['PATH'] = str(args.ads_root/'bin')+os.pathsep+env['PATH']
    network_rows, checks, power_rows = [], [], []
    provenance = {'engine_sha256': digest(engine), 'inputs': [], 'method': 'Saved-current far-field postprocessing only'}
    for case in CASES:
        original = source/'cases'/case
        work = output/('work_'+case)
        work.mkdir(exist_ok=True)
        for file in original.iterdir():
            if file.is_file():
                shutil.copy2(file, work/file.name)
        out = output/case
        out.mkdir(exist_ok=True)
        dsfile = source/'data'/('Crossed_'+case+'.ds')
        inputs = {str(file.relative_to(args.proposal_root)): digest(file)
                  for file in [dsfile, original/'proj.msf', original/'proj.mmd', original/'proj.ltd', original/'proj.prt']}
        provenance['inputs'].append(inputs)
        cache_manifest = out/'cache_manifest.json'
        manifest = {'inputs': inputs, 'engine': provenance['engine_sha256']}
        if cache_manifest.exists() and json.loads(cache_manifest.read_text()) != manifest:
            raise RuntimeError('Cached fields have different inputs; choose a fresh output directory')
        cache_manifest.write_text(json.dumps(manifest, indent=2))
        with dataset.open_dataset_for_reading(dsfile) as ds:
            frame = ds['data'].to_dataframe()
        saved = list(csv.DictReader((original/'results/loaded_impedance.csv').open()))
        for f in frame.index:
            f = float(f)
            s = np.array([[complex(frame.at[f, f'S[{a+1},{b+1}]']) for b in range(4)] for a in range(4)])
            assert all(complex(frame.at[f, f'PORTZ[{a+1}]']) == 50 for a in range(4))
            z = 50*np.linalg.solve((np.eye(4)-s).T, (np.eye(4)+s).T).T
            resistance = (z+z.conj().T)/2
            load = 1/(1e-6+2j*np.pi*f*35e-12)
            L = np.linalg.solve((z+load*np.eye(4)).T, load*np.eye(4)).T
            errs, fieldcols, currentcols = [], [], []
            for a in range(4):
                rest = [b for b in range(4) if b != a]
                zeff = z[a,a]-z[a,rest]@np.linalg.solve(z[np.ix_(rest,rest)]+load*np.eye(3), z[rest,a])
                old = next(r for r in saved if float(r['frequency_mhz']) == f/1e6 and int(r['active_port']) == a+1)
                errs.append(abs(zeff-complex(float(old['loaded_real_ohm']), float(old['loaded_imag_ohm']))))
                for b in range(4):
                    network_rows.append(dict(case=case, frequency_mhz=f/1e6, row=a+1, column=b+1,
                        s_real=s[a,b].real, s_imag=s[a,b].imag, z_real_ohm=z[a,b].real,
                        z_imag_ohm=z[a,b].imag, load_real=L[a,b].real, load_imag=L[a,b].imag))
                (angles, normfield), report, zs = run_fields(work, out, engine, env, f, a, load)
                vs = np.full(4, 1e-12, complex); vs[a] = 1
                currents = np.linalg.solve(z+np.diag(zs), vs)
                voltage = z@currents
                pnet = float(np.vdot(currents, voltage).real/2)
                fieldcols.append(normfield*report['emax_v'])
                currentcols.append(currents)
                power_rows.append(dict(case=case, frequency_mhz=f/1e6, port=a+1,
                    accepted_power_peak_phasor_w=pnet, **report,
                    reported_prad_over_paccepted=report['radiated_power_w']/pnet))
            # G[angle, theta/phi, port] maps peak port current to r*E in vacuum.
            F = np.stack(fieldcols, axis=-1)
            I = np.column_stack(currentcols)
            G = np.linalg.solve(I.T, F.reshape(-1,4).T).T.reshape(F.shape)
            upper = angles[:,0] <= 90
            aa, gg = angles[upper], G[upper]
            weights = np.sin(np.deg2rad(aa[:,0]))*np.deg2rad(1)**2
            weights[aa[:,0] == 90] *= 0.5
            # Receive covariance is transpose of the partial transmit-power matrix.
            rsky = np.einsum('dap,daq,d->pq', gg, gg.conj(), weights)/ETA
            residual = resistance-rsky
            hoc = (4*np.pi/(1j*(2*np.pi*f/C)*ETA))*gg.transpose(0,2,1)
            hload = np.einsum('pq,dqa->dpa', L, hoc)
            zp = np.linalg.solve(np.linalg.inv(z)+np.eye(4)/load, np.eye(4))
            # Provisional split: all non-sky dissipation at uniform 100 K; PEC conductors.
            tsky = 50000*(f/20e6)**-2.5
            sky_psd = 4*KB*tsky*(L@rsky@L.conj().T)
            loss_psd = 4*KB*100*(L@residual@L.conj().T)
            rec_psd = (3e-9)**2*np.eye(4)+4*KB*300/1e6*(zp@zp.conj().T)
            np.savez_compressed(out/f'response_f{f/1e6:06.2f}.npz', frequency_hz=f, angles_deg=aa,
                z_ohm=z, loading=L, hoc_candidate_m=hoc, hload_candidate_m=hload,
                sky_resistance_candidate_ohm=rsky, residual_resistance_candidate_ohm=residual,
                sky_psd_candidate_v2_hz=sky_psd, loss_psd_candidate_v2_hz=loss_psd,
                receiver_psd_v2_hz=rec_psd)
            checks.append(dict(case=case, frequency_mhz=f/1e6,
                reciprocity_relative=float(np.linalg.norm(z-z.T)/np.linalg.norm(z)),
                minimum_network_resistance_ohm=float(np.linalg.eigvalsh(resistance).min()),
                maximum_singular_s=float(np.linalg.svd(s,compute_uv=False).max()),
                maximum_saved_impedance_error_ohm=max(errs),
                minimum_residual_resistance_ohm=float(np.linalg.eigvalsh(residual).min()),
                max_sky_resistance_fraction=float(np.max(np.diag(rsky).real/np.diag(resistance).real)),
                current_basis_condition=float(np.linalg.cond(I))))
            print(case, f/1e6, checks[-1], flush=True)
    write_csv(output/'complex_network.csv', network_rows)
    write_csv(output/'normalization_checks.csv', checks)
    write_csv(output/'port_power_audit.csv', power_rows)
    provenance['checks'] = checks
    provenance['status'] = 'Candidate field normalization; requires independent reciprocity/energy validation before science use'
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    main()
