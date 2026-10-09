"""Test corrected retrieval uncertainty and deliberately out-of-basis injections.

These are controlled counterexamples, not new EM regolith simulations.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PROPOSAL = ROOT.parents[2]/'external/em_archive'
CODE = PROPOSAL/'ads_momentum_benchmark/run_regolith_retrieval_study.py'
spec = importlib.util.spec_from_file_location('retrieval', CODE)
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def main():
    feature = PROPOSAL/'ads_momentum_benchmark/output/crossed_station_pilot/optimized_station_pca_features.csv'
    names, matrix = r.response_matrix(r.read_csv(feature))
    modes, scales, _, _ = r.mode_basis(matrix)
    f = np.linspace(15,25,101)
    foreground, _ = r.foreground_model(f)
    noise = ((foreground+300)/np.sqrt(1e5*1000*3600)).ravel()
    dense = np.array([r.interpolate_anchor_vector(mode,f) for mode in modes])
    signal = r.signal_template(f).reshape(2,-1)
    x = np.log(f/20)
    fourth = (f/20)**-2.5*x**4
    fourth *= 0.1/np.max(abs(fourth))
    ripple = np.sin(2*np.pi*(f-15)/2)
    between = np.sin(np.pi*(f-15)/5)
    assert np.max(abs(between[[0,50,100]])) < 1e-14
    cases = [('null', 0., foreground.copy(), 'Exact training family, no signal'),
             ('matched_40mK', .04, foreground+.04*signal, 'Exact training family'),
             ('foreground_fourth_order', .04, foreground+.04*signal+fourth,
              'Unmodeled smooth term with 0.1 K maximum amplitude in each channel'),
             ('foreground_ripple', .04, foreground+.04*signal+.1*ripple,
              'Unmodeled 0.1 K sinusoid, period 2 MHz, identical in both channels'),
             ('inter_anchor_response', .04, (foreground+.04*signal)*(1+1e-5*between),
              'Fractional response 1e-5 sin(pi*(nu-15)/5), zero at all three EM anchors'),
             ('shifted_signal', .04, foreground-.04*np.exp(-.5*((f-21)/1.5)**2),
              'True trough at 21 MHz fitted with fixed 20 MHz template; shape mismatch, not like-for-like amplitude')]
    rows = []
    for name, amplitude, data, explanation in cases:
        fit = r.fit_spectrum(data.ravel(),noise,f,foreground,dense,scales,3,None)
        rows.append(dict(case=name, injected_amplitude_mk=amplitude*1000,
                         fitted_amplitude_mk=fit['signal_amplitude_k']*1000,
                         amplitude_difference_mk=(fit['signal_amplitude_k']-amplitude)*1000,
                         sigma_mk=fit['signal_sigma_k']*1000, rank=fit['retained_rank'],
                         residual_rms_mk=fit['residual_rms_k']*1000, interpretation=explanation))
    # Independent random data realizations exercise the estimator, not just its formula.
    rng = np.random.default_rng(20261008)
    coverage = []
    for name in ('null','matched_40mK','inter_anchor_response'):
        _, amplitude, data, _ = next(c for c in cases if c[0] == name)
        values = []
        for _ in range(2000):
            fit = r.fit_spectrum(data.ravel()+rng.normal(size=noise.size)*noise,
                                 noise,f,foreground,dense,scales,3,None)
            values.append(fit['signal_amplitude_k'])
        values = np.array(values); sigma = fit['signal_sigma_k']
        coverage.append(dict(case=name, draws=len(values),
            empirical_std_mk=float(values.std(ddof=1)*1000), predicted_sigma_mk=sigma*1000,
            mean_difference_mk=float((values.mean()-amplitude)*1000),
            coverage_68=float(np.mean(abs(values-amplitude) <= sigma)),
            coverage_95=float(np.mean(abs(values-amplitude) <= 1.96*sigma))))
    for row in coverage[:2]:
        assert abs(row['empirical_std_mk']/row['predicted_sigma_mk']-1)<.06
        assert .93 < row['coverage_95'] < .97
    # The corrected fitter must reject truly non-identifiable parameterizations.
    try:
        r.fit_spectrum(foreground.ravel(),noise,f,foreground,np.zeros_like(dense),scales,3,None)
    except ValueError:
        rank_check = True
    else:
        raise AssertionError('Rank-deficient uncertainty was accepted')
    out = ROOT/'results/retrieval_corrected'
    out.mkdir(exist_ok=True)
    payload = dict(status='Controlled model-mismatch stress tests; not independent EM validation',
                   fit_script_sha256=hashlib.sha256(CODE.read_bytes()).hexdigest(),
                   features_sha256=hashlib.sha256(feature.read_bytes()).hexdigest(),
                   rank_deficient_fit_rejected=rank_check, deterministic=rows, monte_carlo=coverage,
                   limitations=['Same five-profile, three-anchor training ensemble.',
                                'Uniform scalar channel noise, no correlated calibration noise.',
                                'Signal-shape mismatch amplitude is a template projection.'])
    (out/'stress_validation.json').write_text(json.dumps(payload,indent=2)+'\n')
    r.write_csv(out/'stress_injections.csv',rows)
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
