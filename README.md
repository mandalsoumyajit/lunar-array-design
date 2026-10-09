# Lunar low-frequency array: design-trade models

Code and processed data for the paper

> B. R. B. Saliwanchik, N. Chakraborty, J. C. Estrada Vigil, S. Herrmann, S. Mandal, L. Philip, A. Slosar, P. Stankus and E. Tamura, *A Deployable Lunar Low-Frequency Radio Array: Design Trades and a Ground-Test Strategy* (submitted to PASP).

The package reproduces the paper's data figures and quoted numbers. It covers:

- the coupled four-port station response and its numerical checks;
- the receiver noise budget;
- the isolated upper-band dipole comparison;
- the worked two-station visibility example;
- the array-layout comparison;
- fiber timing correction and free-space optical (FSO) link closure;
- the scalar spectral-inference stress test.

All results are numerical studies of specified designs, not hardware measurements.

## Requirements

Python ≥ 3.10 with `numpy`, `scipy` and `matplotlib` (see `requirements.txt`).

Two steps also need Keysight ADS 2026 with its Python interpreter:
- the electromagnetic solves, done with the planar method-of-moments solver Momentum;
- reading native Momentum datasets.

The processed station responses in this repository let every downstream study run without ADS.

## Reproducing the figures

Run from the repository root.

| Paper item | Script | ADS needed? |
|---|---|---|
| Fig. 1 (ARES spectrum), Fig. 4 (station response), Fig. 7 (timing/FSO), Fig. 9 (inference test) | `science_model/plot_paper_results.py` | no |
| FSO closure table used by Fig. 7(c) | `science_model/fso_exact_capture.py` (run first) | no |
| Receiver study (62,400-case sweep, stress and temperature screens) | `science_model/receiver_requirements.py` | no |
| Fig. 5 (receiver), Fig. 8 (layouts), Table 3 (visibility example), tolerance tests | `science_model/paper_design_decisions.py` | no |
| Causal timing check | `science_model/causal_timing_check.py` | no |
| Fig. 6 (upper-band dipole) | `science_model/summarize_high_band.py` | yes (verifies the native Momentum solves) |
| Station EM solves and power/receive validation | `physical_station_solver.py`, `export_physical_response.py`, `validate_physical_station.py`, `validate_physical_bridge.py` | yes |

The station-response loader normally re-verifies the raw Momentum solves behind each response. Those solves are not in this repository (see below). Set `LUNAR_ARRAY_PROCESSED_DATA=1` to use the shipped, hash-checked responses instead; the loader still checks the response, solution and validation hashes and the recorded power-closure result.

```bash
export LUNAR_ARRAY_PROCESSED_DATA=1
python science_model/fso_exact_capture.py
python science_model/receiver_requirements.py
python science_model/plot_paper_results.py
python science_model/paper_design_decisions.py
python science_model/causal_timing_check.py
```

With ADS available, `science_model/build_paper_figures.py` runs the whole chain in order. It checks every recorded input hash and writes `science_model/results/paper_evidence/figure_validation.json`.

Different matplotlib versions change the figure rendering slightly but not the data. Regenerated CSV outputs match the committed ones to 1e-9.

## Key modelling choices

- **Receiver criterion:** receiver-added noise (amplifier voltage and current noise, bias resistors and series protection) must stay below 0.25 of the sky noise in both HF receptor modes, after full four-port propagation.
- **Ground emission:** reported separately as `ground_over_sky`, with a 250 K default brightness and a 200–260 K screen. At 14–46 MHz the regolith radiates from below the diurnal thermal skin depth, where the temperature is nearly constant.
- **Sky model:** uniform, T_sky = s · 50,000 (ν/20 MHz)^−2.5 K, with s = 0.3, 1 and 3.
- **Station geometry:** PEC crossed strips, 15.10 m at 6 m height (LF) and 6.05 m at 0.5 m height (HF). They sit over a 1 m layer (ε_r = 3.4, tan δ = 0.005) on a half-space (ε_r = 4.4, tan δ = 0.01), with 13 frequency samples from 1 to 50 MHz.
- **FSO closure:** exact offset-Gaussian aperture capture, with the beam radius including its waist. This replaces the small-aperture approximation of the archived proposal script `results/paper_evidence/simulate_fso_link_availability.py`.

## Layout

```
science_model/                 analysis and validation scripts
science_model/results/
  power_fix/station/physical_lossy_f*/     checked station responses (response.npz) + power/receive validation
  power_fix/station/physical_layered_f*/   lossless-ground validation solutions
  receiver_requirements/       receiver sweep, screens and validation
  design_decisions/            receiver boundary, tolerances, visibility example, layouts, timing check
  high_band_validation/        upper-band summary, per-mesh metrics, solved currents, solve manifests
  paper_evidence/              ARES spectrum, timing/FSO tables, retained proposal inputs, figure manifest
  retrieval_corrected/         scalar inference stress test
figures/                       paper figures (PNG) and TikZ diagrams
```

`response.npz` files hold:
- the 4×4 impedance matrix `z_ohm`;
- sky and ground radiation-resistance matrices;
- open-circuit and loaded effective lengths `hoc_m` / `hload_m`, with shape (direction, port, θ/φ) in metres;
- direction grids `angles_deg` (zenith angle, azimuth).

The phase origin is the ground surface at the station centre.
Their `ground_psd_v2_hz` and `receiver_psd_v2_hz` fields are legacy illustrative exports for a 100 K ground and a 35 pF receiver. The paper's noise results come from `receiver_requirements.py`.

## Not included

The raw Momentum projects, substrate caches, diagnostic kernels and rejected cases (about 2.7 GB) are omitted. They are too large for GitHub and are needed only to rerun the native EM solves.

For publication, local path prefixes in the provenance records were replaced by package-relative paths, with the raw EM archive referred to as `external/em_archive`. The SHA-256 records of the affected files were recomputed so that every hash referring to a file in this repository matches it. Hashes of files that are not shipped are kept as recorded.

## License

Copyright (c) 2026, Brookhaven Science Associates, LLC.

Code is released under the BSD 3-Clause License (`LICENSE`). Data, figures and documentation are released under CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the paper if you use this material.
