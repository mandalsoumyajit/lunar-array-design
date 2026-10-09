"""Explicit, gated bridge from four-port EM recovery to two-receptor RIME.

The recovered antenna fields have NOT passed broadband energy validation.
Default use is rejected. An explicit diagnostic opt-in permits integration tests.
"""
import json
from pathlib import Path
import numpy as np
from measurement_model import sky_covariance, visibility_covariance

ROOT = Path(__file__).resolve().parent
KB = 1.380649e-23
ETA = 376.730313668


def load_station_response(path, ports=(2, 3), *, allow_unvalidated=False):
    if not allow_unvalidated:
        raise ValueError('Recovered fields have unresolved energy validation; diagnostic opt-in required')
    if len(ports) != 2 or len(set(ports)) != 2 or any(p not in range(4) for p in ports):
        raise ValueError('Select two distinct zero-based ports from the four-port station')
    with np.load(path) as d:
        # Select only AFTER complete four-port loading and noise propagation.
        h = d['hload_candidate_m'][:, ports, :]
        jones = h*np.sqrt(ETA*1e-26/2)
        noise4 = d['loss_psd_candidate_v2_hz']+d['receiver_psd_v2_hz']
        return d['angles_deg'].copy(), jones, noise4[np.ix_(ports, ports)], float(d['frequency_hz'])


def load_physical_station_response(path, ports=(2, 3)):
    """Load only the repaired model after power and direct-receive checks.

    This gate certifies numerical consistency of the declared ideal model;
    it does not certify a measured antenna or an absolute science forecast.
    """
    from validate_physical_station import verify_case
    from recover_response import digest
    path=Path(path);directory=path.parent
    if path.name!='response.npz':raise ValueError('Expected corrected response.npz')
    if len(ports)!=2 or len(set(ports))!=2 or any(p not in range(4) for p in ports):
        raise ValueError('Select two distinct zero-based ports from the four-port station')
    record=verify_case(directory)
    report=json.loads((directory/'response_validation.json').read_text())
    if not record['lossy'] or not report['passed_numerical_response_checks']:
        raise ValueError('Corrected response has not passed the required checks')
    if digest(directory/'validation.json')!=report['power_validation_sha256']:
        raise ValueError('Power validation changed after response export')
    if digest(path)!=report['response_sha256']:
        raise ValueError('Response changed after validation')
    if digest(ROOT/'export_physical_response.py')!=report['exporter_sha256']:
        raise ValueError('Response exporter changed; regenerate response')
    with np.load(path) as d:
        h=d['hload_m'][:,ports,:]
        noise=d['ground_psd_v2_hz']+d['receiver_psd_v2_hz']
        return d['angles_deg'].copy(),h*np.sqrt(ETA*1e-26/2),noise[np.ix_(ports,ports)],float(d['frequency_hz'])


def main():
    file = ROOT/'results/response_recovery/lf6m_hf0p5m_rot000_lfband/response_f020.00.npz'
    try:
        load_station_response(file)
    except ValueError:
        gate = True
    else:
        raise AssertionError('Unvalidated response escaped the gate')
    angles, jones, noise, frequency = load_station_response(file, allow_unvalidated=True)
    i = np.flatnonzero((angles[:,0] == 30)&(angles[:,1] == 60))[0]
    th, ph = np.deg2rad(angles[i])
    source = np.array([[np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)]])
    xy = np.array([[0,0,0],[40,0,0],[0,50,0],[60,70,0]], float)
    jj = np.broadcast_to(jones[i], (len(xy),1,2,2))
    response = sky_covariance(xy, source, [[1,0,0,0]], frequency, jones=jj)
    injected = 1000*response
    pairs = [(2*p,2*q) for p in range(4) for q in range(p+1,4)]
    forward = np.array([response[a,b] for a,b in pairs])
    data = np.array([injected[a,b] for a,b in pairs])
    recovered = float(np.vdot(forward,data).real/np.vdot(forward,forward).real)
    assert np.isclose(recovered,1000)
    # Independent thermal equilibrium identity for passive antenna + bias loads.
    with np.load(file) as d:
        z, L = d['z_ohm'], d['loading']
    zl = 1/(1e-6+2j*np.pi*frequency*35e-12)
    zp = np.linalg.inv(np.linalg.inv(z)+np.eye(4)/zl)
    r = (z+z.conj().T)/2
    equilibrium = 4*KB*300*(L@r@L.conj().T+(zp@zp.conj().T)/1e6)
    expected = 4*KB*300*(zp+zp.conj().T)/2
    error = float(np.linalg.norm(equilibrium-expected)/np.linalg.norm(expected))
    assert error < 1e-10
    total = injected.copy()
    for p in range(4):
        total[2*p:2*p+2,2*p:2*p+2] += noise
    cov, relation = visibility_covariance(total,pairs,1e5)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12*np.linalg.norm(cov)
    payload = dict(status='Diagnostic integration test only; not a sensitivity or imaging-performance demonstration',
                   default_gate_rejects_unvalidated_fields=gate,
                   selected_ports_zero_based=[2,3], injected_flux_jy=1000,
                   recovered_noiseless_flux_jy=recovered,
                   thermal_equilibrium_relative_error=error,
                   covariance_positive=True,
                   remaining='Diffuse sky with interstation covariance, independently validated receive fields, site rotation and an imaging estimator')
    (ROOT/'results/response_recovery/bridge_validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
