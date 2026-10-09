"""FSO link-closure sweep with exact offset-Gaussian aperture capture.

Replaces the small-aperture pointing-loss approximation used in the retained
proposal script results/paper_evidence/simulate_fso_link_availability.py,
which is invalid when the beam at the receiver is smaller than the aperture
(short ranges).  Terminal parameters are imported unchanged from that script.

Model: Gaussian beam with 1/e^2 far-field half-angle divergence theta, waist
w0 = lambda/(pi theta) and radius w(z) = w0 sqrt(1 + (z/zR)^2).  A transmitter
pointing error theta_e displaces the beam centre by r = z theta_e at the
receiver.  The fraction of power inside a circular aperture of radius a is the
CDF of a noncentral chi-square with 2 degrees of freedom:
    F = ncx2.cdf((a/s)^2, 2, (r/s)^2),  s = w/2.
Closure means received power >= receiver sensitivity.
"""
from pathlib import Path
import csv, json, sys
import numpy as np
from scipy.stats import ncx2

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'results/paper_evidence'
sys.path.insert(0, str(EVIDENCE))
import simulate_fso_link_availability as old  # noqa: E402

SEED = 20261009
DRAWS = 40_000
OUT = EVIDENCE / 'fso_link_budget_exact.csv'


def capture_fraction(range_m, divergence_rad, offset_m, aperture_d=old.RECEIVER_DIAMETER_M):
    w0 = old.WAVELENGTH_M / (np.pi * divergence_rad)
    zr = np.pi * w0**2 / old.WAVELENGTH_M
    w = w0 * np.sqrt(1.0 + (range_m / zr) ** 2)
    s = w / 2.0
    a = aperture_d / 2.0
    return np.clip(ncx2.cdf((a / s) ** 2, 2, (np.asarray(offset_m) / s) ** 2), 1e-30, 1.0)


def margin_db(range_m, divergence_rad, pointing_rad, dust_db, background_db=0.0):
    frac = capture_fraction(range_m, divergence_rad, range_m * np.asarray(pointing_rad))
    received = old.TX_POWER_DBM - old.FIXED_OPTICAL_LOSS_DB + 10 * np.log10(frac) - dust_db
    return received - old.RECEIVER_SENSITIVITY_DBM - background_db


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for range_km in (0.1, 1.0, 10.0):
        for div_mrad in (0.10, 0.25, 0.50, 1.00):
            for dust_db in (0.0, 3.0, 6.0, 10.0):
                for background_db in (0.0, 3.0, 6.0):
                    for rms in (25.0, 50.0, 100.0, 150.0):
                        err = np.hypot(rng.normal(0, rms * 1e-6, DRAWS), rng.normal(0, rms * 1e-6, DRAWS))
                        m = margin_db(range_km * 1e3, div_mrad * 1e-3, err, dust_db, background_db)
                        rows.append(dict(range_km=range_km, divergence_mrad=div_mrad, dust_loss_db=dust_db,
                                         background_penalty_db=background_db, pointing_axis_rms_urad=rms,
                                         nominal_margin_db=float(margin_db(range_km * 1e3, div_mrad * 1e-3, 0.0, dust_db, background_db)),
                                         closure_probability=float(np.mean(m >= 0)),
                                         margin_p05_db=float(np.quantile(m, .05)),
                                         margin_median_db=float(np.quantile(m, .5))))
    with OUT.open('w', newline='', encoding='utf-8') as h:
        wr = csv.DictWriter(h, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    # Checks: (1) on-axis exact capture reduces to 1-exp(-2a^2/w^2);
    # (2) for a << w, offset loss matches the small-aperture law exp(-2 r^2/w^2).
    w0 = old.WAVELENGTH_M / (np.pi * 1e-3); zr = np.pi * w0**2 / old.WAVELENGTH_M
    w = w0 * np.sqrt(1 + (1e4 / zr) ** 2); a = old.RECEIVER_DIAMETER_M / 2
    on_axis = capture_fraction(1e4, 1e-3, 0.0)
    assert abs(on_axis / (1 - np.exp(-2 * a**2 / w**2)) - 1) < 1e-6
    r = 0.5 * w
    small = capture_fraction(1e4, 1e-3, r) / on_axis
    assert abs(small / np.exp(-2 * r**2 / w**2) - 1) < 1e-3
    sel = {(x['range_km'], x['divergence_mrad']): x['closure_probability'] for x in rows
           if x['dust_loss_db'] == 6 and x['background_penalty_db'] == 0 and x['pointing_axis_rms_urad'] == 100}
    summary = dict(seed=SEED, draws=DRAWS, checks_passed=True,
                   closure_dust6_point100={f'{k[0]}km_{k[1]}mrad': v for k, v in sel.items()})
    (EVIDENCE / 'fso_link_budget_exact.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
