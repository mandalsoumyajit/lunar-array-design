"""Stage 3 free-space optical (FSO) timing-link screening simulation.

This proposal-oriented model deliberately separates two kinds of input:

1. physics-based Gaussian-beam aperture capture and pointing loss; and
2. declared operational stress scenarios for blockage, reacquisition, and dust.

The outage rates and duration distributions are not measurements of the lunar
environment.  They are transparent stress cases used to translate link
availability into oscillator-holdover and data-validity requirements.  Replace
them with breadboard, chamber, and field measurements as the work matures.

The timing link is assumed to discipline a locally recording digitizer.  Science
data can therefore remain valid during an optical outage only while the local
oscillator remains inside the 50 ps inter-cluster timing allocation.  If the
FSO link also carries the only science-data path, its communications availability
must be budgeted separately.
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
TIMING_REQUIREMENT_PS = 50.0
OBSERVATION_HOURS = 24.0
MONTE_CARLO_TRIALS = 20_000

# Representative terminal assumptions.  These are exposed screening values,
# not a selected flight design or a receiver-vendor claim.
TX_POWER_DBM = 10.0
FIXED_OPTICAL_LOSS_DB = 6.0
RECEIVER_SENSITIVITY_DBM = -30.0
RECEIVER_DIAMETER_M = 0.10
WAVELENGTH_M = 1550e-9
BASELINE_RANGE_M = 1_000.0
BASELINE_DIVERGENCE_RAD = 0.50e-3  # Gaussian 1/e^2 intensity half-angle.
CALIBRATION_RECOVERY_S = 30.0


@dataclass(frozen=True)
class StressScenario:
    """Declared blockage/reacquisition scenario, conditional on link closure."""

    name: str
    short_name: str
    local_rate_per_hour: float
    local_block_median_s: float
    local_reacquire_median_s: float
    common_rate_per_hour: float
    common_block_median_s: float
    common_reacquire_median_s: float
    persistent_dust_db: float
    residual_pointing_rms_urad: float
    interpretation: str


SCENARIOS = (
    StressScenario(
        name="Quiet science",
        short_name="quiet",
        local_rate_per_hour=0.20,
        local_block_median_s=2.0,
        local_reacquire_median_s=3.0,
        common_rate_per_hour=0.02,
        common_block_median_s=15.0,
        common_reacquire_median_s=5.0,
        persistent_dust_db=3.0,
        residual_pointing_rms_urad=50.0,
        interpretation="Static stations; rare local interruptions and rare shared obscuration.",
    ),
    StressScenario(
        name="Nearby rover operations",
        short_name="rover",
        local_rate_per_hour=1.0,
        local_block_median_s=8.0,
        local_reacquire_median_s=8.0,
        common_rate_per_hour=0.10,
        common_block_median_s=30.0,
        common_reacquire_median_s=10.0,
        persistent_dust_db=6.0,
        residual_pointing_rms_urad=100.0,
        interpretation="Repeated local obstruction plus occasional shared dust/line-of-sight event.",
    ),
    StressScenario(
        name="Deployment / plume stress",
        short_name="plume",
        local_rate_per_hour=2.0,
        local_block_median_s=20.0,
        local_reacquire_median_s=15.0,
        common_rate_per_hour=0.30,
        common_block_median_s=60.0,
        common_reacquire_median_s=20.0,
        persistent_dust_db=10.0,
        residual_pointing_rms_urad=150.0,
        interpretation="Deliberate high-stress case, not a forecast of normal lunar operations.",
    ),
)

OSCILLATORS = {
    r"$|y|=10^{-11}$": 1e-11,
    r"$|y|=10^{-12}$": 1e-12,
    r"$|y|=10^{-13}$": 1e-13,
}


def gaussian_capture_fraction(
    range_m: np.ndarray | float,
    divergence_rad: np.ndarray | float,
    receiver_diameter_m: np.ndarray | float = RECEIVER_DIAMETER_M,
) -> np.ndarray:
    """On-axis circular-aperture capture for a Gaussian far-field beam."""
    beam_radius_m = np.asarray(range_m) * np.asarray(divergence_rad)
    aperture_radius_m = 0.5 * np.asarray(receiver_diameter_m)
    exponent = -2.0 * (aperture_radius_m / beam_radius_m) ** 2
    return np.clip(1.0 - np.exp(exponent), 1e-15, 1.0)


def pointing_loss_db(
    pointing_error_rad: np.ndarray | float,
    divergence_rad: np.ndarray | float,
) -> np.ndarray:
    """Small-aperture Gaussian pointing loss in dB."""
    ratio = np.asarray(pointing_error_rad) / np.asarray(divergence_rad)
    return (20.0 / np.log(10.0)) * ratio**2


def received_power_dbm(
    range_m: np.ndarray | float,
    divergence_rad: np.ndarray | float,
    pointing_error_rad: np.ndarray | float = 0.0,
    dust_loss_db: np.ndarray | float = 0.0,
    receiver_diameter_m: np.ndarray | float = RECEIVER_DIAMETER_M,
) -> np.ndarray:
    """Screening received power for the representative optical terminal."""
    capture = gaussian_capture_fraction(
        range_m, divergence_rad, receiver_diameter_m
    )
    return (
        TX_POWER_DBM
        - FIXED_OPTICAL_LOSS_DB
        + 10.0 * np.log10(capture)
        - pointing_loss_db(pointing_error_rad, divergence_rad)
        - np.asarray(dust_loss_db)
    )


def link_margin_db(
    *args: object,
    background_penalty_db: np.ndarray | float = 0.0,
    **kwargs: object,
) -> np.ndarray:
    """Received power relative to sensitivity, including a background penalty."""
    return (
        received_power_dbm(*args, **kwargs)
        - RECEIVER_SENSITIVITY_DBM
        - np.asarray(background_penalty_db)
    )


def merge_intervals(
    intervals: list[tuple[float, float]], observation_s: float
) -> list[tuple[float, float]]:
    """Clip, sort, and merge time intervals."""
    clipped = [
        (max(0.0, start), min(observation_s, end))
        for start, end in intervals
        if end > 0.0 and start < observation_s and end > start
    ]
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return intersections between two sorted, merged interval lists."""
    intersections: list[tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            intersections.append((start, end))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return intersections


def lognormal_duration(
    rng: np.random.Generator, median_s: float, log_sigma: float, count: int
) -> np.ndarray:
    """Positive duration distribution specified by its median."""
    if count == 0:
        return np.empty(0)
    return rng.lognormal(mean=np.log(median_s), sigma=log_sigma, size=count)


def generate_path_intervals(
    rng: np.random.Generator,
    observation_s: float,
    rate_per_hour: float,
    block_median_s: float,
    reacquire_median_s: float,
) -> list[tuple[float, float]]:
    """Poisson event times plus lognormal blockage and reacquisition durations."""
    event_count = rng.poisson(rate_per_hour * observation_s / 3600.0)
    if event_count == 0:
        return []
    starts = rng.uniform(0.0, observation_s, size=event_count)
    blockage = lognormal_duration(rng, block_median_s, 0.80, event_count)
    reacquisition = lognormal_duration(rng, reacquire_median_s, 0.50, event_count)
    intervals = list(zip(starts, starts + blockage + reacquisition, strict=True))
    return merge_intervals(intervals, observation_s)


def system_outages(
    rng: np.random.Generator,
    scenario: StressScenario,
    topology: str,
    observation_s: float,
) -> list[tuple[float, float]]:
    """Generate single-path or two-path system outages.

    A two-path timing system is unavailable only during a shared/common event or
    while independent local outages overlap.  The shared-event term prevents an
    unrealistic claim that adding a second optical path removes dust plumes,
    common line-of-sight loss, power loss, or terminal-wide faults.
    """
    common = generate_path_intervals(
        rng,
        observation_s,
        scenario.common_rate_per_hour,
        scenario.common_block_median_s,
        scenario.common_reacquire_median_s,
    )
    local_a = generate_path_intervals(
        rng,
        observation_s,
        scenario.local_rate_per_hour,
        scenario.local_block_median_s,
        scenario.local_reacquire_median_s,
    )
    if topology == "single":
        return merge_intervals(common + local_a, observation_s)
    if topology != "dual":
        raise ValueError(f"Unknown topology: {topology}")
    local_b = generate_path_intervals(
        rng,
        observation_s,
        scenario.local_rate_per_hour,
        scenario.local_block_median_s,
        scenario.local_reacquire_median_s,
    )
    simultaneous_local = intersect_intervals(local_a, local_b)
    return merge_intervals(common + simultaneous_local, observation_s)


def invalid_intervals_after_holdover(
    outages: list[tuple[float, float]],
    maximum_holdover_s: float,
    observation_s: float,
) -> list[tuple[float, float]]:
    """Intervals outside the timing budget, including recovery after failure."""
    invalid = []
    for start, end in outages:
        if end - start > maximum_holdover_s:
            invalid.append(
                (
                    start + maximum_holdover_s,
                    min(observation_s, end + CALIBRATION_RECOVERY_S),
                )
            )
    return merge_intervals(invalid, observation_s)


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in intervals))


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def run_link_sweep(
    output_dir: Path, rng: np.random.Generator
) -> list[dict[str, float]]:
    """Monte Carlo link closure table across range/divergence/dust/pointing."""
    ranges_km = (0.1, 1.0, 10.0)
    divergences_mrad = (0.10, 0.25, 0.50, 1.00)
    dust_losses_db = (0.0, 3.0, 6.0, 10.0)
    background_penalties_db = (0.0, 3.0, 6.0)
    pointing_rms_urad = (25.0, 50.0, 100.0, 150.0)
    draws = 40_000
    rows: list[dict[str, float]] = []
    for range_km in ranges_km:
        for divergence_mrad in divergences_mrad:
            for dust_db in dust_losses_db:
                for background_db in background_penalties_db:
                    for pointing_rms in pointing_rms_urad:
                        # Rayleigh magnitude from independent azimuth/elevation errors.
                        dx = rng.normal(0.0, pointing_rms * 1e-6, size=draws)
                        dy = rng.normal(0.0, pointing_rms * 1e-6, size=draws)
                        pointing = np.hypot(dx, dy)
                        margin = link_margin_db(
                            range_km * 1e3,
                            divergence_mrad * 1e-3,
                            pointing,
                            dust_db,
                            background_penalty_db=background_db,
                        )
                        nominal = float(
                            link_margin_db(
                                range_km * 1e3,
                                divergence_mrad * 1e-3,
                                0.0,
                                dust_db,
                                background_penalty_db=background_db,
                            )
                        )
                        rows.append(
                            {
                                "range_km": range_km,
                                "divergence_mrad": divergence_mrad,
                                "dust_loss_db": dust_db,
                                "background_penalty_db": background_db,
                                "pointing_axis_rms_urad": pointing_rms,
                                "nominal_margin_db": nominal,
                                "closure_probability": float(np.mean(margin >= 0.0)),
                                "margin_p05_db": quantile(margin, 0.05),
                                "margin_median_db": quantile(margin, 0.50),
                            }
                        )
    write_dict_csv(output_dir / "fso_link_budget_sweep.csv", rows)
    return rows


def run_outage_monte_carlo(
    output_dir: Path, rng: np.random.Generator, trials: int
) -> tuple[list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    """Simulate continuous outages, dual-path benefit, and timing holdover."""
    observation_s = OBSERVATION_HOURS * 3600.0
    summaries: list[dict[str, object]] = []
    longest_samples: dict[tuple[str, str], np.ndarray] = {}

    for scenario in SCENARIOS:
        for topology in ("single", "dual"):
            longest = np.zeros(trials)
            downtime_fraction = np.zeros(trials)
            event_count = np.zeros(trials)
            durations_by_trial: list[np.ndarray] = []
            outage_sets: list[list[tuple[float, float]]] = []

            for trial in range(trials):
                outages = system_outages(rng, scenario, topology, observation_s)
                outage_sets.append(outages)
                durations = np.asarray(
                    [end - start for start, end in outages], dtype=float
                )
                durations_by_trial.append(durations)
                event_count[trial] = durations.size
                if durations.size:
                    longest[trial] = float(np.max(durations))
                    downtime_fraction[trial] = float(np.sum(durations) / observation_s)

            longest_samples[(scenario.short_name, topology)] = longest

            for oscillator_label, fractional_frequency_error in OSCILLATORS.items():
                maximum_holdover_s = (
                    TIMING_REQUIREMENT_PS * 1e-12 / fractional_frequency_error
                )
                invalid_fraction = np.zeros(trials)
                trial_holdover_fraction = np.ones(trials)
                all_durations: list[float] = []
                all_success: list[bool] = []
                maximum_error_ps = np.zeros(trials)

                for trial, (outages, durations) in enumerate(
                    zip(outage_sets, durations_by_trial, strict=True)
                ):
                    if durations.size:
                        successes = durations <= maximum_holdover_s
                        trial_holdover_fraction[trial] = float(np.mean(successes))
                        all_durations.extend(durations.tolist())
                        all_success.extend(successes.tolist())
                        maximum_error_ps[trial] = float(
                            np.max(durations)
                            * fractional_frequency_error
                            * 1e12
                        )
                    invalid = invalid_intervals_after_holdover(
                        outages, maximum_holdover_s, observation_s
                    )
                    invalid_fraction[trial] = (
                        interval_duration(invalid) / observation_s
                    )

                event_holdover_success = (
                    float(np.mean(all_success)) if all_success else 1.0
                )
                summaries.append(
                    {
                        "scenario": scenario.name,
                        "scenario_key": scenario.short_name,
                        "topology": topology,
                        "oscillator_label": oscillator_label,
                        "fractional_frequency_error": fractional_frequency_error,
                        "holdover_limit_s": maximum_holdover_s,
                        "trials": trials,
                        "observation_h": OBSERVATION_HOURS,
                        "mean_system_outages_per_day": float(np.mean(event_count)),
                        "downtime_fraction_mean": float(np.mean(downtime_fraction)),
                        "downtime_fraction_p95": quantile(downtime_fraction, 0.95),
                        "longest_outage_p50_s": quantile(longest, 0.50),
                        "longest_outage_p95_s": quantile(longest, 0.95),
                        "longest_outage_p99_s": quantile(longest, 0.99),
                        "event_holdover_success_probability": event_holdover_success,
                        "observation_holdover_success_probability": float(
                            np.mean(invalid_fraction == 0.0)
                        ),
                        "trial_holdover_fraction_p05": quantile(
                            trial_holdover_fraction, 0.05
                        ),
                        "timing_invalid_fraction_mean": float(
                            np.mean(invalid_fraction)
                        ),
                        "timing_invalid_fraction_p95": quantile(
                            invalid_fraction, 0.95
                        ),
                        "maximum_time_error_p95_ps": quantile(
                            maximum_error_ps, 0.95
                        ),
                    }
                )

    write_dict_csv(output_dir / "fso_outage_monte_carlo.csv", summaries)
    return summaries, longest_samples


def minimum_receiver_diameter(
    range_m: np.ndarray,
    divergence_rad: float,
    required_margin_db: float,
    dust_loss_db: float,
) -> np.ndarray:
    """Invert the on-axis Gaussian capture equation for receiver diameter."""
    required_capture_db = (
        RECEIVER_SENSITIVITY_DBM
        + required_margin_db
        - TX_POWER_DBM
        + FIXED_OPTICAL_LOSS_DB
        + dust_loss_db
    )
    required_capture = np.clip(10.0 ** (required_capture_db / 10.0), 1e-15, 0.999)
    beam_radius = range_m * divergence_rad
    radius = beam_radius * np.sqrt(-0.5 * np.log(1.0 - required_capture))
    return 2.0 * radius


def make_link_trade_figure(output_dir: Path) -> None:
    """Physics-based FSO sizing and pointing trade figure."""
    ranges_km = np.logspace(-1, 1.15, 350)
    divergences_mrad = (0.10, 0.25, 0.50, 1.00)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(divergences_mrad)))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)

    ax = axes[0, 0]
    for divergence_mrad, color in zip(divergences_mrad, colors, strict=True):
        margin = link_margin_db(
            ranges_km * 1e3,
            divergence_mrad * 1e-3,
            0.0,
            6.0,
        )
        ax.semilogx(
            ranges_km,
            margin,
            color=color,
            label=f"{divergence_mrad:.2f} mrad",
        )
    ax.axhline(0.0, color="black", lw=1.0)
    ax.axhline(10.0, color="0.45", lw=1.0, ls="--")
    ax.set(
        xlabel="Range (km)",
        ylabel="Optical margin (dB)",
        title="(a) 10 cm receiver, 6 dB dust stress",
        ylim=(-20, 45),
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Half-angle", fontsize=8)

    ax = axes[0, 1]
    normalized_error = np.linspace(0.0, 2.0, 300)
    ax.plot(
        normalized_error,
        pointing_loss_db(normalized_error, 1.0),
        color="#c44e52",
        lw=2.0,
    )
    for value in (0.5, 1.0):
        ax.axvline(value, color="0.45", lw=0.9, ls="--")
    ax.set(
        xlabel=r"Residual pointing error / beam half-angle, $\delta/\theta$",
        ylabel="Gaussian pointing loss (dB)",
        title="(b) Pointing penalty",
        xlim=(0, 2),
        ylim=(0, 36),
    )
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    for divergence_mrad, color in zip(divergences_mrad, colors, strict=True):
        margin_without_pointing = link_margin_db(
            ranges_km * 1e3,
            divergence_mrad * 1e-3,
            0.0,
            6.0,
        )
        allowable_loss_db = np.maximum(margin_without_pointing, 0.0)
        allowable_pointing_urad = (
            divergence_mrad
            * 1e3
            * np.sqrt(allowable_loss_db * np.log(10.0) / 20.0)
        )
        allowable_pointing_urad[margin_without_pointing <= 0.0] = np.nan
        ax.semilogx(
            ranges_km,
            allowable_pointing_urad,
            color=color,
            label=f"{divergence_mrad:.2f} mrad",
        )
    ax.set(
        xlabel="Range (km)",
        ylabel=r"Maximum residual pointing error ($\mu$rad)",
        title="(c) Zero-margin pointing boundary",
        ylim=(0, 1400),
    )
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 1]
    for divergence_mrad, color in zip(divergences_mrad, colors, strict=True):
        diameter = minimum_receiver_diameter(
            ranges_km * 1e3,
            divergence_mrad * 1e-3,
            required_margin_db=10.0,
            dust_loss_db=6.0,
        )
        ax.loglog(
            ranges_km,
            diameter,
            color=color,
            label=f"{divergence_mrad:.2f} mrad",
        )
    ax.axhline(RECEIVER_DIAMETER_M, color="black", lw=1.0, ls="--")
    ax.text(0.11, 0.112, "10 cm reference", fontsize=8, va="bottom")
    ax.set(
        xlabel="Range (km)",
        ylabel="Required receiver diameter (m)",
        title="(d) Aperture for 10 dB margin",
        ylim=(0.008, 2.0),
    )
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Stage 3 FSO screening link budget: "
        f"{TX_POWER_DBM:.0f} dBm launch, {FIXED_OPTICAL_LOSS_DB:.0f} dB fixed loss, "
        f"{RECEIVER_SENSITIVITY_DBM:.0f} dBm sensitivity",
        fontsize=11,
    )
    save_figure(fig, output_dir / "fso_link_pointing_trade")


def make_outage_figure(
    output_dir: Path,
    summaries: list[dict[str, object]],
    longest_samples: dict[tuple[str, str], np.ndarray],
) -> None:
    """Continuous-outage distributions and timing-validity consequences."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.9), constrained_layout=True)
    scenario_colors = {
        "quiet": "#4c78a8",
        "rover": "#f58518",
        "plume": "#b33c86",
    }

    ax = axes[0, 0]
    for scenario in SCENARIOS:
        for topology, linestyle in (("single", "-"), ("dual", "--")):
            samples = np.sort(longest_samples[(scenario.short_name, topology)])
            cdf = np.arange(1, samples.size + 1) / samples.size
            positive = samples > 0.0
            ax.semilogx(
                samples[positive],
                cdf[positive],
                color=scenario_colors[scenario.short_name],
                ls=linestyle,
                label=f"{scenario.name}, {topology}",
            )
    ax.axvline(5.0, color="0.35", lw=0.9, ls=":")
    ax.axvline(50.0, color="0.35", lw=0.9, ls=":")
    ax.axvline(500.0, color="0.35", lw=0.9, ls=":")
    ax.set(
        xlabel="Longest continuous outage in 24 h (s)",
        ylabel="Cumulative probability",
        title="(a) Outage duration, not just availability",
        xlim=(0.5, 1200),
        ylim=(0, 1.01),
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    ax = axes[0, 1]
    x = np.arange(len(SCENARIOS))
    width = 0.34
    for offset, topology, color in (
        (-width / 2, "single", "#5b8ff9"),
        (width / 2, "dual", "#61d9a8"),
    ):
        values = []
        for scenario in SCENARIOS:
            row = next(
                row
                for row in summaries
                if row["scenario_key"] == scenario.short_name
                and row["topology"] == topology
                and float(row["fractional_frequency_error"]) == 1e-12
            )
            values.append(float(row["longest_outage_p95_s"]))
        ax.bar(x + offset, values, width=width, label=topology, color=color)
    ax.axhline(50.0, color="black", lw=1.0, ls="--", label=r"50 s: $10^{-12}$")
    ax.set_xticks(x, ["Quiet", "Rover", "Plume"])
    ax.set(
        ylabel="95th-percentile longest outage (s)",
        title="(b) Redundancy retains common-mode outages",
        yscale="log",
        ylim=(1, 1000),
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rover_rows = [
        row
        for row in summaries
        if row["scenario_key"] == "rover"
    ]
    oscillator_values = list(OSCILLATORS.values())
    labels = [r"$10^{-11}$", r"$10^{-12}$", r"$10^{-13}$"]
    x = np.arange(len(oscillator_values))
    for offset, topology, color in (
        (-width / 2, "single", "#5b8ff9"),
        (width / 2, "dual", "#61d9a8"),
    ):
        values = []
        for oscillator in oscillator_values:
            row = next(
                row
                for row in rover_rows
                if row["topology"] == topology
                and float(row["fractional_frequency_error"]) == oscillator
            )
            values.append(
                100.0 * float(row["observation_holdover_success_probability"])
            )
        ax.bar(x + offset, values, width=width, label=topology, color=color)
    ax.set_xticks(x, labels)
    ax.set(
        xlabel=r"Assumed $|y|$ during holdover",
        ylabel="24 h observations without violation (%)",
        title="(c) Nearby-rover timing-budget survival",
        ylim=(0, 105),
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    markers = {"single": "o", "dual": "s"}
    for scenario in SCENARIOS:
        for topology in ("single", "dual"):
            values = []
            for oscillator in oscillator_values:
                row = next(
                    row
                    for row in summaries
                    if row["scenario_key"] == scenario.short_name
                    and row["topology"] == topology
                    and float(row["fractional_frequency_error"]) == oscillator
                )
                values.append(
                    100.0 * float(row["timing_invalid_fraction_mean"])
                )
            ax.plot(
                oscillator_values,
                values,
                color=scenario_colors[scenario.short_name],
                marker=markers[topology],
                ls="-" if topology == "single" else "--",
                label=f"{scenario.name}, {topology}",
            )
    ax.set(
        xlabel=r"Assumed holdover fractional-frequency error, $|y|$",
        ylabel="Mean timing-invalid data (%)",
        title="(d) Local recording; 30 s recovery after violation",
        xscale="log",
        yscale="log",
        xlim=(2e-11, 5e-14),
        ylim=(1e-5, 3),
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Stage 3 scenario Monte Carlo (20,000 independent 24 h observations)",
        fontsize=11,
    )
    save_figure(fig, output_dir / "fso_outage_holdover_trade")


def make_representative_timeline(
    output_dir: Path, rng: np.random.Generator
) -> None:
    """Generate a human-readable rover-operations example at one-second cadence."""
    scenario = next(item for item in SCENARIOS if item.short_name == "rover")
    observation_s = 6.0 * 3600.0
    time_s = np.arange(0.0, observation_s + 1.0, 1.0)
    time_h = time_s / 3600.0

    # Residual pointing combines a slow thermal boresight term with jitter.
    thermal_x = 70e-6 * np.sin(2.0 * np.pi * time_h / 4.0)
    thermal_y = 45e-6 * np.sin(2.0 * np.pi * time_h / 2.7 + 0.8)
    jitter_x = rng.normal(0.0, 30e-6, size=time_s.size)
    jitter_y = rng.normal(0.0, 30e-6, size=time_s.size)
    pointing = np.hypot(thermal_x + jitter_x, thermal_y + jitter_y)
    dust_db = 3.0 + 3.0 * time_s / observation_s
    margin = link_margin_db(
        BASELINE_RANGE_M,
        BASELINE_DIVERGENCE_RAD,
        pointing,
        dust_db,
    )

    # Select a valid stress-scenario realization that includes a budget-crossing
    # outage so the figure demonstrates the holdover state transition visibly.
    # This is illustrative and is not used in the unbiased Monte Carlo summary.
    outages: list[tuple[float, float]] = []
    for _ in range(200):
        candidate = system_outages(rng, scenario, "single", observation_s)
        longest = max((end - start for start, end in candidate), default=0.0)
        outages = candidate
        if 60.0 <= longest <= 180.0:
            break
    blocked = np.zeros(time_s.size, dtype=bool)
    outage_age_s = np.zeros(time_s.size)
    for start, end in outages:
        indices = (time_s >= start) & (time_s <= end)
        blocked[indices] = True
        outage_age_s[indices] = time_s[indices] - start
    effective_margin = margin.copy()
    effective_margin[blocked] = -35.0
    holdover_error_ps = outage_age_s * 1e-12 * 1e12

    rows = [
        {
            "time_s": float(time_s[index]),
            "time_h": float(time_h[index]),
            "pointing_error_urad": float(pointing[index] * 1e6),
            "dust_loss_db": float(dust_db[index]),
            "unblocked_margin_db": float(margin[index]),
            "blocked": int(blocked[index]),
            "outage_age_s": float(outage_age_s[index]),
            "holdover_error_ps_at_y_1e-12": float(holdover_error_ps[index]),
        }
        for index in range(0, time_s.size, 5)
    ]
    write_dict_csv(output_dir / "fso_representative_timeline.csv", rows)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.7), sharex=True, constrained_layout=True)
    axes[0].plot(time_h, margin, color="#4c78a8", lw=0.7, label="unblocked margin")
    axes[0].fill_between(
        time_h,
        -40,
        35,
        where=blocked,
        color="#d62728",
        alpha=0.24,
        label="blockage + reacquisition",
    )
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set(
        ylabel="Margin (dB)",
        title="(a) 1 km / 0.5 mrad representative optical state",
        ylim=(-8, 25),
    )
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_h, pointing * 1e6, color="#f58518", lw=0.7)
    axes[1].set(
        ylabel=r"Pointing error ($\mu$rad)",
        title="(b) Declared thermal drift plus residual jitter",
    )
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_h, holdover_error_ps, color="#b33c86", lw=1.0)
    axes[2].axhline(
        TIMING_REQUIREMENT_PS,
        color="black",
        lw=1.0,
        ls="--",
        label="50 ps requirement",
    )
    axes[2].set(
        xlabel="Elapsed time (h)",
        ylabel="Holdover error (ps)",
        title=r"(c) Optimistic constant $|y|=10^{-12}$ holdover",
        ylim=(0, max(70.0, 1.05 * float(np.max(holdover_error_ps)))),
    )
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Illustrative FSO timing timeline; selected stress realization, not a lunar prediction",
        fontsize=11,
    )
    save_figure(fig, output_dir / "fso_representative_timeline")


def write_assumptions(output_dir: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "parameter": "random seed",
            "value": SEED,
            "units": "",
            "classification": "reproducibility",
            "notes": "Fixed for all Stage 3 outputs.",
        },
        {
            "parameter": "wavelength",
            "value": WAVELENGTH_M * 1e9,
            "units": "nm",
            "classification": "representative design",
            "notes": "Compatible with the published reciprocal FSO White Rabbit demonstration.",
        },
        {
            "parameter": "launch power",
            "value": TX_POWER_DBM,
            "units": "dBm",
            "classification": "assumed terminal",
            "notes": "Replace with measured telescope-output power.",
        },
        {
            "parameter": "fixed optical loss",
            "value": FIXED_OPTICAL_LOSS_DB,
            "units": "dB",
            "classification": "assumed terminal",
            "notes": "Circulators, telescope throughput, contamination-independent losses, and implementation margin.",
        },
        {
            "parameter": "receiver sensitivity",
            "value": RECEIVER_SENSITIVITY_DBM,
            "units": "dBm",
            "classification": "assumed terminal",
            "notes": "Must be replaced by measured packet/timing validity versus received power and background.",
        },
        {
            "parameter": "solar/background sensitivity penalty sweep",
            "value": "0/3/6",
            "units": "dB",
            "classification": "stress assumption",
            "notes": "Lumped receiver-sensitivity degradation; replace with measured background and detector-noise curves.",
        },
        {
            "parameter": "receiver diameter",
            "value": RECEIVER_DIAMETER_M,
            "units": "m",
            "classification": "representative design",
            "notes": "Clear aperture.",
        },
        {
            "parameter": "baseline range",
            "value": BASELINE_RANGE_M / 1e3,
            "units": "km",
            "classification": "stretch case",
            "notes": "Physics plots sweep 0.1--14 km.",
        },
        {
            "parameter": "baseline divergence",
            "value": BASELINE_DIVERGENCE_RAD * 1e3,
            "units": "mrad",
            "classification": "representative design",
            "notes": "Gaussian 1/e^2 intensity half-angle.",
        },
        {
            "parameter": "timing allocation",
            "value": TIMING_REQUIREMENT_PS,
            "units": "ps",
            "classification": "science-derived requirement",
            "notes": "From Stage 1 visibility-coherence analysis.",
        },
        {
            "parameter": "post-violation recovery",
            "value": CALIBRATION_RECOVERY_S,
            "units": "s",
            "classification": "operational assumption",
            "notes": "Data remain invalid until a fresh timing/calibration state is established.",
        },
        {
            "parameter": "Monte Carlo trials",
            "value": MONTE_CARLO_TRIALS,
            "units": "24 h observations per case",
            "classification": "numerical",
            "notes": "Single- and dual-path cases for each stress scenario.",
        },
    ]
    for scenario in SCENARIOS:
        rows.extend(
            [
                {
                    "parameter": f"{scenario.short_name}: local event rate",
                    "value": scenario.local_rate_per_hour,
                    "units": "h^-1 path^-1",
                    "classification": "stress assumption",
                    "notes": scenario.interpretation,
                },
                {
                    "parameter": f"{scenario.short_name}: local block/reacquire medians",
                    "value": f"{scenario.local_block_median_s}/{scenario.local_reacquire_median_s}",
                    "units": "s/s",
                    "classification": "stress assumption",
                    "notes": "Independent between redundant paths.",
                },
                {
                    "parameter": f"{scenario.short_name}: shared event rate",
                    "value": scenario.common_rate_per_hour,
                    "units": "h^-1",
                    "classification": "stress assumption",
                    "notes": "Common-mode obstruction, dust, power, or terminal event.",
                },
                {
                    "parameter": f"{scenario.short_name}: shared block/reacquire medians",
                    "value": f"{scenario.common_block_median_s}/{scenario.common_reacquire_median_s}",
                    "units": "s/s",
                    "classification": "stress assumption",
                    "notes": "Affects both optical paths.",
                },
                {
                    "parameter": f"{scenario.short_name}: persistent dust",
                    "value": scenario.persistent_dust_db,
                    "units": "dB",
                    "classification": "stress assumption",
                    "notes": "Used in link-closure screening; not a lunar attenuation distribution.",
                },
            ]
        )
    write_dict_csv(output_dir / "fso_stage3_assumptions.csv", rows)


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        path_without_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def print_key_results(summaries: list[dict[str, object]]) -> None:
    print("Stage 3 FSO timing-link screening results")
    print(
        f"Terminal: {TX_POWER_DBM:.0f} dBm launch, {RECEIVER_DIAMETER_M*100:.0f} cm receiver, "
        f"{RECEIVER_SENSITIVITY_DBM:.0f} dBm sensitivity, {FIXED_OPTICAL_LOSS_DB:.0f} dB fixed loss"
    )
    for range_km in (1.0, 10.0):
        margin = float(
            link_margin_db(
                range_km * 1e3,
                BASELINE_DIVERGENCE_RAD,
                100e-6,
                6.0,
            )
        )
        print(
            f"  {range_km:g} km, 0.5 mrad, 100 urad residual, 6 dB dust: "
            f"{margin:.2f} dB margin"
        )
    print("\n95th-percentile longest outage and 1e-12 holdover consequences:")
    for scenario in SCENARIOS:
        for topology in ("single", "dual"):
            row = next(
                item
                for item in summaries
                if item["scenario_key"] == scenario.short_name
                and item["topology"] == topology
                and float(item["fractional_frequency_error"]) == 1e-12
            )
            print(
                f"  {scenario.name:25s} {topology:6s}: "
                f"longest p95={float(row['longest_outage_p95_s']):6.1f} s, "
                f"events within 50 ps={100*float(row['event_holdover_success_probability']):5.1f}%, "
                f"mean timing-invalid={100*float(row['timing_invalid_fraction_mean']):.4f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/timing_simulations"),
        help="Directory for proposal-ready figures and CSV outputs.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=MONTE_CARLO_TRIALS,
        help="Independent 24 h observations per scenario and topology.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_assumptions(args.output_dir)
    run_link_sweep(args.output_dir, np.random.default_rng(SEED + 1))
    make_link_trade_figure(args.output_dir)
    summaries, longest_samples = run_outage_monte_carlo(
        args.output_dir, np.random.default_rng(SEED + 2), args.trials
    )
    make_outage_figure(args.output_dir, summaries, longest_samples)
    make_representative_timeline(args.output_dir, np.random.default_rng(SEED + 3))
    print_key_results(summaries)


if __name__ == "__main__":
    main()



