"""Full-polarization point-source RIME and reproducible capability screens.

Run: python science_model/measurement_model.py
Dependency: numpy. Outputs are confined to science_model/results.
This is a reference implementation and analytic sensitivity screen, not an
EM beam prediction, diffuse-sky simulator, or cosmological detection forecast.
"""
from pathlib import Path
import csv
import json
import math
import hashlib
import numpy as np

C = 299792458.0
KB = 1.380649e-23
NU21_MHZ = 1420.40575177
ROOT = Path(__file__).resolve().parent


def coherency(stokes):
    """Jy convention: unpolarized flux S gives Bxx=Byy=S; V sign as in note."""
    i, q, u, v = stokes
    if i < 0 or q*q + u*u + v*v > i*i + 1e-12:
        raise ValueError("Unphysical Stokes vector")
    return np.array([[i+q, u+1j*v], [u-1j*v, i-q]], dtype=complex)


def sky_covariance(positions_m, directions, stokes_jy, frequency_hz,
                   jones=None, delays_s=None):
    """Return (2N,2N) covariance, node-major X,Y, for discrete far-field sources.

    Directions must be unit vectors in the local Cartesian frame. Jones has
    shape (N,S,2,2), maps sky polarization to calibrated receptors, and excludes
    the geometric phase and scalar clock delay applied here. Occulted sources
    must be omitted or assigned zero Jones response. No noise or mutual
    coupling is silently added; use embedded loaded beams for a real array.
    """
    r = np.asarray(positions_m, dtype=float)
    s = np.asarray(directions, dtype=float)
    b = np.asarray(stokes_jy, dtype=float)
    n, ns = len(r), len(s)
    if r.shape != (n, 3) or s.shape != (ns, 3) or b.shape != (ns, 4):
        raise ValueError("Inconsistent sky/position shapes")
    if not np.allclose(np.linalg.norm(s, axis=1), 1):
        raise ValueError("Directions must be unit vectors")
    if jones is None:
        jones = np.broadcast_to(np.eye(2), (n, ns, 2, 2))
    jones = np.asarray(jones, dtype=complex)
    if jones.shape != (n, ns, 2, 2):
        raise ValueError("Jones shape must be N,S,2,2")
    delays = np.zeros(n) if delays_s is None else np.asarray(delays_s)
    if delays.shape != (n,):
        raise ValueError("Delay shape must be N")
    phase = np.exp(2j*np.pi*frequency_hz*(delays[:, None] - r@s.T/C))
    out = np.zeros((2*n, 2*n), dtype=complex)
    for k in range(ns):
        a = (phase[:, k, None, None]*jones[:, k]).reshape(2*n, 2)
        out += a @ coherency(b[k]) @ a.conj().T
    return out


def visibility_covariance(r, pairs, independent_samples):
    """Complex Gaussian sample-covariance errors, including shared-sky noise.

    Returns conjugated covariance E[dV dV^H] and relation E[dV dV^T].
    For overlapping channelizer samples, K requires an effective sample count.
    """
    if independent_samples <= 0:
        raise ValueError("Independent sample count must be positive")
    cov = np.empty((len(pairs), len(pairs)), dtype=complex)
    rel = cov.copy()
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            cov[a, b] = r[i, k]*r[l, j]/independent_samples
            rel[a, b] = r[i, l]*r[k, j]/independent_samples
    return cov, rel


def reference_response(frequency_mhz, cfg):
    """Toy per-receptor area and temperature; never inferred from sky fraction."""
    lam = C/(frequency_mhz*1e6)
    eta = cfg['radiation_efficiency']
    ae = eta*cfg['reference_directivity']*lam**2/(4*np.pi)
    tsky = cfg['sky_temperature_at_20mhz_k']*(frequency_mhz/20)**cfg['sky_spectral_index']
    tsys = eta*tsky + (1-eta)*cfg['regolith_temperature_k'] + cfg['receiver_temperature_k']
    sefd = 2*KB*tsys/ae/1e-26
    return ae, tsys, sefd


def point_rms(sefd, n, bandwidth, time, npol=2):
    return sefd/np.sqrt(npol*n*(n-1)*bandwidth*time)


def distance_and_y(z, cfg):
    """Flat illustrative cosmology, radiation included; distances in Mpc."""
    co = cfg['cosmology']
    om, rad, h0 = co['omega_m'], co['omega_r'], co['H0_km_s_mpc']
    zz = np.linspace(0, z, 20001)
    e = np.sqrt(om*(1+zz)**3+rad*(1+zz)**4+1-om-rad)
    dm = (C/1000)/h0*np.trapezoid(1/e, zz)
    y = (C/1000)*(1+z)**2/(h0*e[-1]*NU21_MHZ*1e6)
    return dm, y


def spiral(n, radius, start=0):
    k = np.arange(n)+start
    angle = k*np.pi*(3-np.sqrt(5))
    radii = radius*np.sqrt((np.arange(n)+0.5)/n)
    return np.column_stack([radii*np.cos(angle), radii*np.sin(angle), np.zeros(n)])


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def validate(cfg):
    """Physical invariants and independent numerical checks of key conventions."""
    checks = {}
    r = np.array([[0.,0,0], [37.,12,1], [-12.,29,0.5]])
    s = np.array([[0.,0,1], [0.6,0,0.8]])
    st = np.array([[8.,1.,2.,-1.], [3.,0,0,0]])
    rr = sky_covariance(r, s, st, 20e6)
    assert np.allclose(rr, rr.conj().T)
    assert np.linalg.eigvalsh(rr).min() > -1e-10
    assert np.allclose(rr, sky_covariance(r+[10,20,30], s, st, 20e6))
    assert np.allclose(rr, sky_covariance(r, s, st, 20e6, delays_s=np.full(3, 9e-9)))
    checks['hermiticity_positivity_translation_common_clock'] = 'passed'
    assert np.allclose(rr[:2,:2], coherency(st.sum(axis=0)))
    checks['stokes_normalization'] = 'passed'
    delta = np.array([50e-12,0,0])
    rc = sky_covariance(r, s[:1], st[:1], 125e6, delays_s=delta)
    r0 = sky_covariance(r, s[:1], st[:1], 125e6)
    assert np.allclose(rc[0,2]/r0[0,2], np.exp(2j*np.pi*125e6*50e-12))
    assert np.allclose(abs(rc[0,2]), abs(r0[0,2]))
    checks['constant_delay_phase_without_decorrelation'] = 'passed'
    rg = r.copy(); rg[0] += C*delta[0]*s[0]
    assert np.allclose(sky_covariance(rg, s[:1], st[:1],125e6,delays_s=delta), r0)
    checks['single_direction_geometry_clock_degeneracy'] = 'passed'
    rng = np.random.default_rng(cfg['random_seed'])
    jitter = rng.normal(0, 50e-12, 500000)
    measured = abs(np.mean(np.exp(2j*np.pi*125e6*jitter)))
    analytic = np.exp(-0.5*(2*np.pi*125e6*50e-12)**2)
    assert abs(measured-analytic) < 8e-6
    checks['gaussian_jitter_coherence'] = {'numerical':measured,'analytic':float(analytic)}
    sefd=1e5; k=3600*1e5
    cv, rel = visibility_covariance(np.eye(4)*sefd, [(0,2)], k)
    assert np.isclose(np.sqrt(0.5*(cv[0,0]+rel[0,0]).real), sefd/np.sqrt(2*k))
    # With correlated sky, both covariance and relation terms are needed.
    truth = np.array([[2, 0.4+0.2j],[0.4-0.2j, 3]], complex)
    zz=(rng.normal(size=(12000,64,2))+1j*rng.normal(size=(12000,64,2)))/np.sqrt(2)
    xx=zz@np.linalg.cholesky(truth).T
    vv=np.mean(xx[:,:,0]*xx[:,:,1].conj(),axis=1)-truth[0,1]
    cc, pp=visibility_covariance(truth,[(0,1)],64)
    assert np.isclose(np.mean(abs(vv)**2),cc[0,0].real,rtol=0.04)
    assert np.isclose(np.var(vv.real),0.5*(cc[0,0]+pp[0,0]).real,rtol=0.04)
    checks['radiometer_and_correlated_sky_noise'] = 'passed'
    return checks


def main():
    cfg_path=ROOT/'reference_config.json'
    cfg=json.loads(cfg_path.read_text())
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    checks=validate(cfg)
    sensitivity=[]
    for f in cfg['frequencies_mhz']:
        ae, tsys, sefd=reference_response(f,cfg)
        for n in cfg['node_counts']:
            rms=point_rms(sefd,n,cfg['channel_bandwidth_hz'],cfg['valid_integration_s'],cfg['independent_polarizations'])
            sensitivity.append(dict(frequency_mhz=f,n_dualpol_nodes=n,ae_per_receptor_m2=ae,tsys_k=tsys,
                sefd_per_receptor_jy=sefd,rms_1h_100khz_jy=rms,five_sigma_1s_100khz_jy=5*rms*np.sqrt(3600)))
    write_csv(results/'reference_sensitivity.csv',sensitivity)
    layouts={'compact':spiral(128,500),'extended':spiral(128,5000),
             'core_and_outer':np.vstack([spiral(96,500),spiral(32,5000,start=96)])}
    layout_rows=[]
    for name,r in layouts.items():
        write_csv(results/f'layout_{name}.csv',[dict(node=i,x_m=x,y_m=y,z_m=z) for i,(x,y,z) in enumerate(r)])
        b=np.linalg.norm(r[:,None,:]-r[None,:,:],axis=-1)[np.triu_indices(128,1)]
        for f in [20,50]:
            lam=C/(f*1e6)
            layout_rows.append(dict(layout=name,frequency_mhz=f,n_baselines=len(b),bmin_m=b.min(),bmax_m=b.max(),
                resolution_arcmin=lam/b.max()*180/np.pi*60,baselines_u_le_10=int(np.sum(b/lam<=10)),
                baselines_u_le_30=int(np.sum(b/lam<=30)),baselines_u_le_100=int(np.sum(b/lam<=100))))
    write_csv(results/'layout_comparison.csv',layout_rows)
    cosmos=[]
    for z in [30,50,70,100]:
        f=NU21_MHZ/(1+z); dm,y=distance_and_y(z,cfg); lam=C/(f*1e6)
        cosmos.append(dict(redshift=z,frequency_mhz=f,transverse_distance_mpc=dm,y_mpc_per_mhz=y*1e6,
            baseline_for_kperp_0p01_m=0.01*dm*lam/(2*np.pi),baseline_for_kperp_0p1_m=0.1*dm*lam/(2*np.pi),
            radial_mpc_per_100khz=y*1e5))
    write_csv(results/'cosmological_scales.csv',cosmos)
    # Rectangular-channel sinc attenuation: solve sinc(x)=0.99, x=B*tau.
    low, high = 0.0, 0.5
    for _ in range(60):
        mid = (low+high)/2
        if np.sinc(mid)>0.99: low=mid
        else: high=mid
    xlim=(low+high)/2
    smearing=[]
    for baseline in [1000,10000]:
        for f in [20,125]:
            omega=2*np.pi/(27.3*86400)  # declared nominal lunar sidereal period
            smearing.append(dict(baseline_m=baseline,frequency_mhz=f,
                channel_hz_for_1pct_zenith_to_horizon=xlim*C/baseline,
                integration_s_for_1pct_max_sidereal_fringe=xlim*C/(f*1e6*baseline*omega)))
    write_csv(results/'smearing_limits.csv',smearing)
    summary={'status':cfg['status'],'config_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'numpy_version':np.__version__,'checks':checks,
        'timing_125mhz':{'fifty_ps_coherence_loss_percent':100*(1-np.exp(-0.5*(2*np.pi*125e6*50e-12)**2)),
            'one_cm_phase_deg':360*125e6*0.01/C,'one_cm_delay_ps':0.01/C*1e12},
        'raw_transport_example':{'sample_rate_msps':300,'bits_per_sample':8,'two_receptors_gbps':4.8,'four_receptors_gbps':9.6},
        'limitations':['Ideal reference collecting area, not loaded EM response.',
            'Point-source sensitivity assumes independent weak-source noise, natural weights, and two usable polarizations.',
            'No confusion, scattering, foreground-calibration floor, RFI, mutual coupling, or site visibility loss in scalar sensitivity.',
            'Layout baseline counts are zenith snapshot geometric counts, not uv dwell time or independent cosmological modes.',
            'Point-source RIME and covariance implemented; diffuse sky, channelizer and end-to-end science estimators remain future extensions.']}
    (results/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'validation':checks,'N128_sensitivity':[a for a in sensitivity if a['n_dualpol_nodes']==128],
                      'layouts':layout_rows,'cosmological_scales':cosmos},indent=2))


if __name__=='__main__':
    main()
