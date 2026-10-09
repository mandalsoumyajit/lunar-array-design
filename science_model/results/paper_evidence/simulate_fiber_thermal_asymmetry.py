"""Stage 2 fiber thermal/asymmetry simulations for the lunar interferometer.

The model separates four effects that are often conflated:

1. absolute reciprocal fiber delay, which White Rabbit (WR) measures through
   the round-trip exchange;
2. residual wavelength-dependent fiber asymmetry;
3. SFP wavelength drift mapped through chromatic dispersion; and
4. endpoint transmit/receive fixed-delay drift.

Reference coefficients come from published measurements and are deliberately
exposed as parameters.  They are not asserted to apply to flight hardware:

* 40 ps/(km K) absolute delay for conventional single-mode fiber;
* -0.2 ps/(km K) WR offset from G.652.D fiber temperature;
* +0.11 and -0.51 ps/(km K) WR offset from two measured SFP configurations;
* 10.85 ps/K endpoint offset, derived from 0.5*(13.3 - (-8.4)) ps/K.

Outputs are proposal-oriented sensitivity figures, cross-validation metrics,
and Monte Carlo tables over the 1--125 MHz science band.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 20260723
SCIENCE_EDGE_HZ = 125e6
REQUIREMENT_PS = 50.0
THERMAL_ALLOCATION_PS = 25.0

ABSOLUTE_FIBER_TCD_PS_PER_KM_K = 40.0
CRYOGENIC_FIBER_TCD_PS_PER_KM_K = 50.9
WR_FIBER_ASYMMETRY_PS_PER_KM_K = -0.20
WR_SFP_COEFFICIENTS_PS_PER_KM_K = {
    "1310-nm FP reference": 0.11,
    "1490-nm DFB reference": -0.51,
}
WR_ENDPOINT_OFFSET_PS_PER_K = 10.85


@dataclass(frozen=True)
class SensorModel:
    cable_lag_minutes: float = 30.0
    sfp_lag_minutes: float = 8.0
    board_lag_minutes: float = 8.0
    cable_noise_k: float = 0.20
    sfp_noise_k: float = 0.10
    board_noise_k: float = 0.10


def coherence_loss_percent(delay_ps: np.ndarray | float) -> np.ndarray:
    """Clock-induced coherence loss at the 125 MHz science-band edge."""
    phase_radians = 2.0 * np.pi * SCIENCE_EDGE_HZ * np.asarray(delay_ps) * 1e-12
    return 100.0 * (1.0 - np.exp(-0.5 * phase_radians**2))


def first_order_sensor(
    true_temperature_k: np.ndarray,
    dt_seconds: float,
    lag_minutes: float,
    noise_k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a first-order thermal lag and additive readout noise."""
    sensed = np.empty_like(true_temperature_k)
    sensed[0] = true_temperature_k[0]
    tau_seconds = max(lag_minutes * 60.0, dt_seconds)
    gain = 1.0 - np.exp(-dt_seconds / tau_seconds)
    for index in range(1, true_temperature_k.size):
        sensed[index] = sensed[index - 1] + gain * (
            true_temperature_k[index] - sensed[index - 1]
        )
    sensed += rng.normal(0.0, noise_k, size=sensed.size)
    return sensed


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def thermal_profiles(time_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent chamber-like cable, SFP, and board temperature cycles."""
    cable_k = (
        240.0
        + 38.0 * np.sin(2.0 * np.pi * time_hours / 15.5)
        + 7.0 * np.sin(2.0 * np.pi * time_hours / 4.7 + 0.4)
    )
    board_k = (
        255.0
        + 18.0 * np.sin(2.0 * np.pi * time_hours / 7.2 + 0.9)
        + 3.0 * np.sin(2.0 * np.pi * time_hours / 2.1)
    )
    sfp_k = (
        252.0
        + 15.0 * np.sin(2.0 * np.pi * time_hours / 5.1 - 0.6)
        + 4.0 * np.sin(2.0 * np.pi * time_hours / 1.8 + 0.2)
    )
    return cable_k, sfp_k, board_k


def simulate_validation_cycle(
    length_km: float,
    sfp_coefficient: float,
    sensor_model: SensorModel,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """Fit on the first thermal cycle and cross-validate on the second."""
    dt_seconds = 60.0
    time_hours = np.arange(0.0, 48.0, dt_seconds / 3600.0)
    cable_k, sfp_k, board_k = thermal_profiles(time_hours)

    cable_sensor_k = first_order_sensor(
        cable_k,
        dt_seconds,
        sensor_model.cable_lag_minutes,
        sensor_model.cable_noise_k,
        rng,
    )
    sfp_sensor_k = first_order_sensor(
        sfp_k,
        dt_seconds,
        sensor_model.sfp_lag_minutes,
        sensor_model.sfp_noise_k,
        rng,
    )
    board_sensor_k = first_order_sensor(
        board_k,
        dt_seconds,
        sensor_model.board_lag_minutes,
        sensor_model.board_noise_k,
        rng,
    )

    reference_k = 250.0
    fiber_offset_ps = (
        WR_FIBER_ASYMMETRY_PS_PER_KM_K
        * length_km
        * (cable_k - reference_k)
    )
    sfp_offset_ps = sfp_coefficient * length_km * (sfp_k - reference_k)
    board_offset_ps = WR_ENDPOINT_OFFSET_PS_PER_K * (board_k - reference_k)
    board_nonlinearity_ps = 0.012 * (board_k - reference_k) ** 2
    wr_precision_ps = rng.normal(0.0, 4.0, size=time_hours.size)
    true_offset_ps = (
        fiber_offset_ps
        + sfp_offset_ps
        + board_offset_ps
        + board_nonlinearity_ps
        + wr_precision_ps
    )

    smooth_samples = int(round(15.0 * 60.0 / dt_seconds))
    cable_smooth = moving_average(cable_sensor_k, smooth_samples)
    sfp_smooth = moving_average(sfp_sensor_k, smooth_samples)
    board_smooth = moving_average(board_sensor_k, smooth_samples)

    cable_delta = cable_smooth - reference_k
    sfp_delta = sfp_smooth - reference_k
    board_delta = board_smooth - reference_k
    cable_rate = np.gradient(cable_smooth, time_hours)
    sfp_rate = np.gradient(sfp_smooth, time_hours)
    board_rate = np.gradient(board_smooth, time_hours)

    static_features = np.column_stack(
        [
            np.ones(time_hours.size),
            length_km * cable_delta,
            length_km * sfp_delta,
            board_delta,
        ]
    )
    lag_aware_features = np.column_stack(
        [
            static_features,
            length_km * cable_rate,
            length_km * sfp_rate,
            board_rate,
            board_delta**2,
        ]
    )

    training = time_hours < 24.0
    validation = ~training

    static_coefficients, *_ = np.linalg.lstsq(
        static_features[training], true_offset_ps[training], rcond=None
    )
    lag_coefficients, *_ = np.linalg.lstsq(
        lag_aware_features[training], true_offset_ps[training], rcond=None
    )

    static_prediction_ps = static_features @ static_coefficients
    lag_prediction_ps = lag_aware_features @ lag_coefficients
    static_residual_ps = true_offset_ps - static_prediction_ps
    lag_residual_ps = true_offset_ps - lag_prediction_ps

    return {
        "time_hours": time_hours,
        "cable_k": cable_k,
        "sfp_k": sfp_k,
        "board_k": board_k,
        "true_offset_ps": true_offset_ps,
        "static_residual_ps": static_residual_ps,
        "lag_residual_ps": lag_residual_ps,
        "training": training,
        "validation": validation,
        "static_coefficients": static_coefficients,
        "lag_coefficients": lag_coefficients,
    }


def residual_metrics(values_ps: np.ndarray) -> dict[str, float]:
    """Timing and 125 MHz science metrics for one residual series."""
    centered = values_ps - np.mean(values_ps)
    absolute = np.abs(centered)
    rms_ps = float(np.sqrt(np.mean(centered**2)))
    return {
        "rms_ps": rms_ps,
        "p95_abs_ps": float(np.percentile(absolute, 95.0)),
        "p99_abs_ps": float(np.percentile(absolute, 99.0)),
        "max_abs_ps": float(np.max(absolute)),
        "phase_rms_deg_125_mhz": 360.0 * SCIENCE_EDGE_HZ * rms_ps * 1e-12,
        "coherence_loss_percent_125_mhz": float(coherence_loss_percent(rms_ps)),
    }


def run_cross_validation(output_dir: Path) -> list[dict[str, float | str]]:
    """Run all length/SFP cases and save a detailed validation figure."""
    rows: list[dict[str, float | str]] = []
    sensor_model = SensorModel()
    plot_case: dict[str, np.ndarray | float] | None = None

    for length_km in (0.1, 1.0, 10.0):
        for sfp_name, sfp_coefficient in WR_SFP_COEFFICIENTS_PS_PER_KM_K.items():
            case_seed = SEED + int(length_km * 100) + (0 if sfp_coefficient > 0 else 1)
            case = simulate_validation_cycle(
                length_km,
                sfp_coefficient,
                sensor_model,
                np.random.default_rng(case_seed),
            )
            validation = np.asarray(case["validation"], dtype=bool)
            for model_name, key in (
                ("uncorrected", "true_offset_ps"),
                ("static-temperature correction", "static_residual_ps"),
                ("lag-aware correction", "lag_residual_ps"),
            ):
                metrics = residual_metrics(np.asarray(case[key])[validation])
                rows.append(
                    {
                        "length_km": length_km,
                        "sfp_reference": sfp_name,
                        "model": model_name,
                        **metrics,
                    }
                )
            if length_km == 10.0 and sfp_coefficient < 0:
                plot_case = case

    with (output_dir / "fiber_cross_validation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if plot_case is None:
        raise RuntimeError("The representative validation case was not generated.")

    time_hours = np.asarray(plot_case["time_hours"])
    validation = np.asarray(plot_case["validation"], dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.0), constrained_layout=True)

    axes[0].plot(time_hours, plot_case["cable_k"], label="Cable", linewidth=1.8)
    axes[0].plot(time_hours, plot_case["sfp_k"], label="SFP", linewidth=1.5)
    axes[0].plot(time_hours, plot_case["board_k"], label="Endpoint board", linewidth=1.5)
    axes[0].axvspan(24.0, 48.0, color="#eef3fa", alpha=0.8, label="Validation interval")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("(a) Independent chamber-like temperature excitation")
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(
        time_hours,
        plot_case["true_offset_ps"],
        color="#b7232a",
        linewidth=1.2,
        label="Uncorrected WR offset",
    )
    axes[1].axhline(REQUIREMENT_PS, color="0.25", linestyle="--", linewidth=1.1)
    axes[1].axhline(-REQUIREMENT_PS, color="0.25", linestyle="--", linewidth=1.1)
    axes[1].axvline(24.0, color="0.3", linestyle=":", linewidth=1.1)
    axes[1].set_ylabel("Offset (ps)")
    axes[1].set_title("(b) 10 km, 1490-nm DFB reference configuration")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        time_hours[validation],
        np.asarray(plot_case["static_residual_ps"])[validation],
        color="#d17c00",
        linewidth=1.0,
        label="Static temperature correction",
    )
    axes[2].plot(
        time_hours[validation],
        np.asarray(plot_case["lag_residual_ps"])[validation],
        color="#0f4c81",
        linewidth=1.0,
        label="Lag-aware correction",
    )
    axes[2].axhline(THERMAL_ALLOCATION_PS, color="0.25", linestyle="--", linewidth=1.1)
    axes[2].axhline(-THERMAL_ALLOCATION_PS, color="0.25", linestyle="--", linewidth=1.1)
    axes[2].set_xlim(24.0, 48.0)
    axes[2].set_xlabel("Elapsed time (h)")
    axes[2].set_ylabel("Validation residual (ps)")
    axes[2].set_title("(c) Frozen-model cross-validation")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        "Stage 2 fiber thermal correction: train on one cycle, validate on the next",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "fiber_thermal_cross_validation")
    return rows


def run_monte_carlo(output_dir: Path, samples: int) -> list[dict[str, float | str | int]]:
    """Propagate coefficient and temperature-telemetry uncertainty."""
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    all_residuals: dict[tuple[float, str], np.ndarray] = {}

    for length_km in (0.1, 1.0, 10.0):
        for sfp_name, sfp_coefficient in WR_SFP_COEFFICIENTS_PS_PER_KM_K.items():
            cable_delta_k = rng.uniform(-100.0, 100.0, samples)
            sfp_delta_k = rng.uniform(-60.0, 60.0, samples)
            board_delta_k = rng.uniform(-60.0, 60.0, samples)

            cable_sensor_error_k = rng.normal(0.0, 0.50, samples)
            sfp_sensor_error_k = rng.normal(0.0, 0.25, samples)
            board_sensor_error_k = rng.normal(0.0, 0.25, samples)

            fiber_coefficient_error = rng.normal(
                0.0, 0.10 * abs(WR_FIBER_ASYMMETRY_PS_PER_KM_K), samples
            )
            sfp_coefficient_error = rng.normal(
                0.0, 0.10 * abs(sfp_coefficient), samples
            )
            board_coefficient_error = rng.normal(0.0, 0.50, samples)

            fiber_component = length_km * (
                fiber_coefficient_error * cable_delta_k
                + WR_FIBER_ASYMMETRY_PS_PER_KM_K * cable_sensor_error_k
            )
            sfp_component = length_km * (
                sfp_coefficient_error * sfp_delta_k
                + sfp_coefficient * sfp_sensor_error_k
            )
            board_component = (
                board_coefficient_error * board_delta_k
                + WR_ENDPOINT_OFFSET_PS_PER_K * board_sensor_error_k
            )
            short_term_precision = rng.normal(0.0, 4.0, samples)
            residual_ps = (
                fiber_component
                + sfp_component
                + board_component
                + short_term_precision
            )
            all_residuals[(length_km, sfp_name)] = residual_ps
            metrics = residual_metrics(residual_ps)
            rows.append(
                {
                    "length_km": length_km,
                    "sfp_reference": sfp_name,
                    "samples": samples,
                    "seed": SEED,
                    **metrics,
                    "pass_50_ps_rms": str(metrics["rms_ps"] <= REQUIREMENT_PS),
                    "pass_25_ps_thermal_allocation": str(
                        metrics["rms_ps"] <= THERMAL_ALLOCATION_PS
                    ),
                }
            )

    with (output_dir / "fiber_uncertainty_monte_carlo.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    colors = ["#0f4c81", "#b7232a"]
    x = np.arange(3)
    width = 0.34
    for index, (sfp_name, _) in enumerate(WR_SFP_COEFFICIENTS_PS_PER_KM_K.items()):
        selected = [row for row in rows if row["sfp_reference"] == sfp_name]
        axes[0].bar(
            x + (index - 0.5) * width,
            [float(row["rms_ps"]) for row in selected],
            width,
            color=colors[index],
            label=sfp_name,
        )
        axes[1].bar(
            x + (index - 0.5) * width,
            [float(row["p95_abs_ps"]) for row in selected],
            width,
            color=colors[index],
            label=sfp_name,
        )

    for axis, title, ylabel in (
        (axes[0], "(a) Residual RMS", "Timing residual RMS (ps)"),
        (axes[1], "(b) 95th-percentile absolute residual", "|Timing residual| (ps)"),
    ):
        axis.axhline(
            THERMAL_ALLOCATION_PS,
            color="#2a8f5b",
            linestyle="--",
            linewidth=1.3,
            label="25 ps thermal allocation",
        )
        axis.axhline(
            REQUIREMENT_PS,
            color="0.3",
            linestyle=":",
            linewidth=1.3,
            label="50 ps total requirement",
        )
        axis.set_xticks(x, ["0.1 km", "1 km", "10 km"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=7.5)
    axes[1].legend(fontsize=7.5)
    fig.suptitle(
        f"Coefficient and telemetry uncertainty Monte Carlo ({samples:,} cases/configuration)",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "fiber_uncertainty_monte_carlo")
    return rows


def save_sensitivity_outputs(output_dir: Path) -> None:
    """Create analytic sensitivity maps and a requirements table."""
    lengths_km = np.logspace(-1.0, 1.0, 300)
    temperature_excursion_k = np.linspace(0.1, 150.0, 350)
    length_grid, temperature_grid = np.meshgrid(lengths_km, temperature_excursion_k)

    absolute_delay_ps = (
        ABSOLUTE_FIBER_TCD_PS_PER_KM_K * length_grid * temperature_grid
    )
    fiber_asymmetry_ps = (
        abs(WR_FIBER_ASYMMETRY_PS_PER_KM_K)
        * length_grid
        * temperature_grid
    )
    sfp_asymmetry_ps = (
        abs(WR_SFP_COEFFICIENTS_PS_PER_KM_K["1490-nm DFB reference"])
        * length_grid
        * temperature_grid
    )
    board_offset_ps = WR_ENDPOINT_OFFSET_PS_PER_K * temperature_excursion_k

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), constrained_layout=True)
    levels = np.logspace(0.0, 6.0, 25)
    contour = axes[0, 0].contourf(
        length_grid,
        temperature_grid,
        absolute_delay_ps,
        levels=levels,
        norm=matplotlib.colors.LogNorm(),
        cmap="Blues",
        extend="both",
    )
    axes[0, 0].contour(
        length_grid,
        temperature_grid,
        absolute_delay_ps,
        levels=[50.0, 1000.0, 10000.0],
        colors=["#2a8f5b", "#d17c00", "#b7232a"],
        linewidths=1.3,
    )
    fig.colorbar(contour, ax=axes[0, 0], label="Uncorrected delay change (ps)")
    axes[0, 0].set_title("(a) Reciprocal fiber delay, 40 ps/(km K)")

    residual_levels = np.linspace(0.0, 800.0, 25)
    residual = axes[0, 1].contourf(
        length_grid,
        temperature_grid,
        fiber_asymmetry_ps,
        levels=residual_levels,
        cmap="YlOrRd",
        extend="max",
    )
    residual_lines = axes[0, 1].contour(
        length_grid,
        temperature_grid,
        fiber_asymmetry_ps,
        levels=[25.0, 50.0, 100.0],
        colors=["#2a8f5b", "#d17c00", "#b7232a"],
        linewidths=1.4,
    )
    axes[0, 1].clabel(residual_lines, fmt=lambda value: f"{value:.0f} ps")
    fig.colorbar(residual, ax=axes[0, 1], label="WR offset change (ps)")
    axes[0, 1].set_title("(b) Fiber asymmetry reference, 0.20 ps/(km K)")

    for coefficient, color, label in (
        (
            abs(WR_FIBER_ASYMMETRY_PS_PER_KM_K),
            "#0f4c81",
            "Fiber-temperature asymmetry",
        ),
        (
            abs(WR_SFP_COEFFICIENTS_PS_PER_KM_K["1490-nm DFB reference"]),
            "#b7232a",
            "1490-nm SFP-temperature reference",
        ),
    ):
        axes[1, 0].plot(
            lengths_km,
            coefficient * lengths_km * 100.0,
            color=color,
            linewidth=2.1,
            label=label,
        )
    axes[1, 0].axhline(
        THERMAL_ALLOCATION_PS,
        color="#2a8f5b",
        linestyle="--",
        linewidth=1.3,
        label="25 ps thermal allocation",
    )
    axes[1, 0].axhline(
        REQUIREMENT_PS,
        color="0.3",
        linestyle=":",
        linewidth=1.3,
        label="50 ps total requirement",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("(c) 100 K excursion: uncorrected nonreciprocity")
    axes[1, 0].set_xlabel("Link length (km)")
    axes[1, 0].set_ylabel("WR offset change (ps)")
    axes[1, 0].legend(fontsize=7.5)

    axes[1, 1].plot(
        temperature_excursion_k,
        board_offset_ps,
        color="#6f42c1",
        linewidth=2.2,
        label="Measured endpoint reference",
    )
    axes[1, 1].plot(
        temperature_excursion_k,
        0.50 * temperature_excursion_k,
        color="#0f4c81",
        linewidth=2.0,
        label="0.5 ps/K coefficient error",
    )
    axes[1, 1].axhline(
        THERMAL_ALLOCATION_PS,
        color="#2a8f5b",
        linestyle="--",
        linewidth=1.3,
        label="25 ps thermal allocation",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(1.0, 3000.0)
    axes[1, 1].set_title("(d) Endpoint fixed-delay sensitivity")
    axes[1, 1].set_xlabel("Endpoint temperature excursion (K)")
    axes[1, 1].set_ylabel("WR offset change (ps)")
    axes[1, 1].legend(fontsize=7.5)

    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.22)
    for axis in axes[0, :]:
        axis.set_xscale("log")
        axis.set_xlabel("Link length (km)")
        axis.set_ylabel("Temperature excursion (K)")

    fig.suptitle(
        "Stage 2 fiber timing sensitivity: reciprocal delay versus residual asymmetry",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "fiber_thermal_sensitivity")

    component_allocation_ps = THERMAL_ALLOCATION_PS / np.sqrt(3.0)
    rows: list[dict[str, float]] = []
    for length_km in (0.1, 1.0, 10.0):
        for sfp_name, sfp_coefficient in WR_SFP_COEFFICIENTS_PS_PER_KM_K.items():
            rows.append(
                {
                    "length_km": length_km,
                    "sfp_coefficient_ps_per_km_k": sfp_coefficient,
                    "raw_delay_for_100_k_ps": ABSOLUTE_FIBER_TCD_PS_PER_KM_K
                    * length_km
                    * 100.0,
                    "fiber_asymmetry_for_100_k_ps": abs(
                        WR_FIBER_ASYMMETRY_PS_PER_KM_K
                    )
                    * length_km
                    * 100.0,
                    "sfp_term_for_60_k_ps": abs(sfp_coefficient)
                    * length_km
                    * 60.0,
                    "fiber_coefficient_1sigma_target_ps_per_km_k": component_allocation_ps
                    / (length_km * 100.0),
                    "sfp_coefficient_1sigma_target_ps_per_km_k": component_allocation_ps
                    / (length_km * 60.0),
                    "fiber_sensor_1sigma_target_k": component_allocation_ps
                    / (abs(WR_FIBER_ASYMMETRY_PS_PER_KM_K) * length_km),
                    "sfp_sensor_1sigma_target_k": component_allocation_ps
                    / (abs(sfp_coefficient) * length_km),
                    "board_coefficient_1sigma_target_ps_per_k": component_allocation_ps
                    / 60.0,
                    "board_sensor_1sigma_target_k": component_allocation_ps
                    / WR_ENDPOINT_OFFSET_PS_PER_K,
                }
            )

    with (output_dir / "fiber_sensitivity_requirements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(
            base_path.with_suffix(f".{extension}"),
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "timing_simulations",
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=300_000,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_sensitivity_outputs(args.output_dir)
    run_cross_validation(args.output_dir)
    run_monte_carlo(args.output_dir, args.monte_carlo_samples)
    print(f"Wrote Stage 2 fiber results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
