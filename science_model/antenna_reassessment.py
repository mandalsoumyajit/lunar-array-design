"""Reanalyse existing ADS terminal impedances; no new electromagnetic solve.

Sweep ONE active receiver while the other three retain 1 Mohm || 35 pF.
These are voltage-transfer/noise screens, not antenna sensitivity forecasts.
Uses only Python's standard library. See ../ANTENNA_REASSESSMENT_2026-10-08.md.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
K_B = 1.380649e-23


def circuit(za, frequency_hz, capacitance_pf, resistance_ohm=1e6):
    zl = 1 / (1 / resistance_ohm + 2j * math.pi * frequency_hz * capacitance_pf * 1e-12)
    transfer = zl / (za + zl)
    parallel = za * transfer
    return transfer, parallel


def noise_oc(za, transfer, parallel, en_nv, in_pa, temperature_k=300, resistance_ohm=1e6):
    # One-sided amplitude spectral densities. Uncorrelated ideal e_n and i_n.
    # Bias resistor noise is separate and must not already be in e_n/i_n.
    amp_psd = (en_nv * 1e-9) ** 2 + abs(parallel) ** 2 * (in_pa * 1e-12) ** 2
    bias_psd = abs(parallel) ** 2 * 4 * K_B * temperature_k / resistance_ohm
    return math.sqrt((amp_psd + bias_psd) / abs(transfer) ** 2) * 1e9


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate():
    # Resistive divider; capacitive divider; open active receiver; current-noise identity.
    a, p = circuit(50 + 0j, 1e6, 0, 50)
    assert abs(a - 0.5) < 1e-14 and abs(p - 25) < 1e-14
    f = 10e6
    ca, cl = 10e-12, 35e-12
    za = 1 / (2j * math.pi * f * ca)
    a, _ = circuit(za, f, cl * 1e12, 1e30)
    assert abs(a - ca / (ca + cl)) < 1e-12
    a, _ = circuit(200 - 900j, f, 0, 1e30)
    assert abs(a - 1) < 1e-12
    for cp in (10, 20, 35, 50):
        za = 80 + 500j
        a, p = circuit(za, f, cp)
        assert abs(p / a - za) < 1e-10
        assert math.isclose(noise_oc(za, a, p, 0, 1, 0), abs(za) * 1e-3, rel_tol=1e-12)
    return ['resistive divider', 'capacitive divider', 'open receiver limit', 'current noise referred to Thevenin source']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-root', type=Path,
                        default=HERE.parent.parent.parent / 'external' / 'em_archive')
    args = parser.parse_args()
    root = args.proposal_root.resolve()
    output = HERE / 'results' / 'antenna_reassessment'
    output.mkdir(parents=True, exist_ok=True)
    checks = validate()
    cases = ['lf6m_hf0p5m_rot000_lfband', 'lf6m_hf0p5m_rot000_highband']
    input_manifest, sweep, screening = [], [], []
    for case in cases:
        path = root / 'ads_momentum_benchmark' / 'output' / 'crossed_station_pilot' / 'cases' / case / 'results' / 'loaded_impedance.csv'
        input_manifest.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            f = float(row['frequency_mhz'])
            za = complex(float(row['loaded_real_ohm']), float(row['loaded_imag_ohm']))
            if za.real < 0:
                raise ValueError('Unexpected negative resistance in source data')
            for cp in (10, 20, 35, 50):
                a, p = circuit(za, f * 1e6, cp)
                common = {'case': case, 'frequency_mhz': f, 'port': row['active_name'],
                          'active_capacitance_pf': cp, 'za_real_ohm': za.real, 'za_imag_ohm': za.imag,
                          'voltage_transfer_abs': abs(a), 'voltage_transfer_db': 20 * math.log10(abs(a)),
                          'voltage_transfer_phase_deg': math.degrees(math.atan2(a.imag, a.real))}
                for en in (1, 3):
                    for inoise in (0, 0.2, 1):
                        eoc = noise_oc(za, a, p, en, inoise)
                        sweep.append({**common, 'en_nv_sqrt_hz': en, 'in_pa_sqrt_hz': inoise,
                                      'receiver_plus_bias_noise_oc_nv_sqrt_hz': eoc,
                                      'sky_oc_asd_for_4_to_1_power_ratio_nv_sqrt_hz': 2 * eoc})
                # Two independent 30 MHz solves remain separate in full CSV.
                # Concise table uses low-band deck through 30 and high-band deck above 30.
                if row['active_name'] in ('LF_X', 'HF_U') and cp in (10, 35) and not (case.endswith('highband') and f == 30):
                    screening.append({**common, 'en_nv_sqrt_hz': 3, 'in_pa_sqrt_hz': 0,
                                      'receiver_plus_bias_noise_oc_nv_sqrt_hz': noise_oc(za, a, p, 3, 0)})
    write_csv(output / 'active_receiver_sweep.csv', sweep)
    write_csv(output / 'selected_loading_comparison.csv', screening)
    electrical = []
    for length in (1.5, 2, 3, 4, 6.05, 15.1):
        for f in (1, 5, 10, 15, 20, 30, 50, 75, 125):
            electrical.append({'tip_to_tip_length_m': length, 'frequency_mhz': f,
                               'length_wavelengths': length * f * 1e6 / 299792458})
    write_csv(output / 'electrical_lengths.csv', electrical)
    manifest = {
        'date': '2026-10-08', 'inputs': input_manifest,
        'checks_passed': checks, 'sweep_rows': len(sweep),
        'assumptions': {
            'active_bias_resistance_ohm': 1e6, 'active_bias_temperature_k': 300,
            'inactive_terminations': '1 Mohm || 35 pF, fixed in source data',
            'voltage_noise_nv_sqrt_hz': [1, 3], 'current_noise_pa_sqrt_hz': [0, 0.2, 1],
            'voltage_current_noise_correlation': 0,
            'not_included': ['sky and loss noise', 'noise emitted by the other three receivers',
                             'changes to other port loads', 'complex effective length or loaded beam',
                             'new antenna geometries', 'LNA gain and stability', 'full-wave solve']},
        'interpretation': 'Voltage transfer is not power gain or sensitivity. Assumed receiver parameters are not measured hardware specifications.'}
    (output / 'validation_and_provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'checks_passed': checks, 'sweep_rows': len(sweep)}, indent=2))
    for row in screening:
        if (row['port'], row['frequency_mhz']) in [('LF_X', 10), ('LF_X', 15), ('LF_X', 20), ('HF_U', 10), ('HF_U', 20), ('HF_U', 30), ('HF_U', 40), ('HF_U', 50)]:
            print(f"{row['port']:4s} {row['frequency_mhz']:4.0f} MHz C={row['active_capacitance_pf']:2d} pF: |A|={row['voltage_transfer_abs']:.4f}, {row['voltage_transfer_db']:.2f} dB, e_oc={row['receiver_plus_bias_noise_oc_nv_sqrt_hz']:.2f} nV/sqrt(Hz)")


if __name__ == '__main__':
    main()
