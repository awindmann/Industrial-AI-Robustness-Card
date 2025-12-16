"""
Human-friendly narrative snippets for the model card dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class NarrativeSection:
    """Named subsection that can carry both text and bullets."""

    key: str
    title: str
    body: str = ""
    bullets: List[str] = field(default_factory=list)
    footer_note: str | None = None


@dataclass(frozen=True)
class NarrativeBlock:
    """Container for one narrative section on the model card."""

    body: str = ""
    bullets: List[str] = field(default_factory=list)
    sections: List[NarrativeSection] = field(default_factory=list)
    footer_note: str | None = None


MODEL_CARD_META: Dict[str, str] = {
    "provider": "Bio Data Science",
    "contact": "penicillin-softsensor@example.com",
}


def load_data_quality_stats(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load data quality statistics from JSON file exported by the data notebook."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "processed" / "indpensim_data_quality.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_data_quality_bullets(stats: Optional[Dict[str, Any]] = None) -> List[str]:
    """Format data quality statistics as bullet points for the model card."""
    if stats is None:
        stats = load_data_quality_stats()
    if stats is None:
        return []

    bullets = []
    raw = stats.get("raw_data", {})
    processed = stats.get("missingness_processed", {})
    repro = stats.get("reproducibility", {})

    # Source context (from paper: 100 kL bioreactor simulation)
    bullets.append("Synthetic fermentation dataset from the IndPenSim benchmark (100 kL bioreactor).")

    # Data characteristics
    if raw.get("total_samples"):
        bullets.append(
            f"Raw data: {raw['total_samples']:,} samples across {raw.get('num_batches', 'N/A')} batches"
        )
    if raw.get("sampling_interval_minutes"):
        bullets.append(
            f"Sampling interval: {raw['sampling_interval_minutes']} minutes ({raw.get('sampling_interval_hours', 0.2):.2f} hours)"
        )
    if raw.get("avg_batch_duration_days"):
        bullets.append(
            f"Average batch duration: {raw['avg_batch_duration_hours']:.1f} hours ({raw['avg_batch_duration_days']:.1f} days)"
        )

    # Missingness
    if processed.get("missing_rate_overall") is not None:
        rate = processed["missing_rate_overall"]
        if rate == 0:
            bullets.append("Missingness after preprocessing: 0% (all features complete)")
        else:
            bullets.append(f"Missingness after preprocessing: {rate:.2f}%")

    # Reproducibility
    if repro.get("random_seed_split"):
        bullets.append(
            f"Reproducibility: seed={repro['random_seed_split']}, Python {repro.get('python_version', 'N/A')}"
        )

    return bullets


# Cached data quality stats for use in MODEL_CARD_TEXT
_DATA_QUALITY_STATS: Optional[Dict[str, Any]] = load_data_quality_stats()
_DATA_QUALITY_BULLETS: List[str] = format_data_quality_bullets(_DATA_QUALITY_STATS)


def format_data_quality_details(stats: Optional[Dict[str, Any]] = None) -> List[str]:
    """Format detailed data quality info (missingness, key feature stats) as bullets."""
    if stats is None:
        stats = load_data_quality_stats()
    if stats is None:
        return []

    bullets = []
    missingness = stats.get("missingness_raw", {})
    processed = stats.get("missingness_processed", {})
    features = stats.get("feature_statistics", {})

    # Missingness summary
    cols_missing = missingness.get("columns_with_missing", 0)
    cols_complete = missingness.get("columns_complete", 0)
    bullets.append(
        f"Raw data: {cols_complete} complete columns, {cols_missing} columns with missing values (offline assays, ~98% missing)."
    )

    # Processed data summary
    proc_features = processed.get("total_features", 0)
    proc_samples = processed.get("total_samples", 0)
    bullets.append(
        f"After preprocessing: {proc_features} features, {proc_samples:,} samples, 0% missing."
    )

    # Key feature statistics (target variable)
    if "penicillin_concentration" in features:
        p = features["penicillin_concentration"]
        bullets.append(
            f"Target (penicillin): mean={p['mean']:.2f}, std={p['std']:.2f}, range=[{p['min']:.2e}, {p['max']:.2f}] g/L."
        )

    # A few key input features
    if "dissolved_oxygen" in features:
        do2 = features["dissolved_oxygen"]
        bullets.append(
            f"Dissolved oxygen: mean={do2['mean']:.1f}, std={do2['std']:.2f}, range=[{do2['min']:.1f}, {do2['max']:.1f}]."
        )

    if "temperature" in features:
        t = features["temperature"]
        bullets.append(
            f"Temperature: mean={t['mean']:.1f}K, std={t['std']:.2f}, range=[{t['min']:.1f}, {t['max']:.1f}]K."
        )

    return bullets


_DATA_QUALITY_DETAILS: List[str] = format_data_quality_details(_DATA_QUALITY_STATS)


TARGET_ALIAS_LABELS = {
    "penicillin": "Penicillin concentration (g/L)",
    "penicillin_concentration": "Penicillin concentration (g/L)",
}


DATASET_TARGET_LABELS = {
    "indpensim": "Penicillin concentration (g/L)",
}


MODEL_CARD_TEXT: Dict[str, NarrativeBlock] = {
    "general_information": NarrativeBlock(
        body=(
            "Data-driven soft sensor estimating penicillin concentration on the IndPenSim benchmark."
        ),
    ),
    "intended_use": NarrativeBlock(
        body=(
            "This model is intended for research and benchmarking on simulated penicillin fermentations."
        ),
        bullets=[
            "Estimates penicillin concentration within a simulated industrial bioreactor.",
            "Supports development and comparison of biopharmaceutical soft-sensor methods.",
            # "Training and evaluation use the IndPenSim simulation of a 100.000l bioreactor acting as a substitute for required laboratory measurement. ",
            "Stress-test scenarios evaluate robustness to sensor noise, missing channels, and injected faults.",
            "Not approved for production deployments or closed-loop actuator control.",
        ],
        # footer_note="Scope: requires validation on real plant data and governance approval before any operational use.",
    ),
    "dataset_spec": NarrativeBlock(
        sections=[
            NarrativeSection(
                key="data-basics",
                title="Data snapshot",
                body="",
                bullets=_DATA_QUALITY_BULLETS if _DATA_QUALITY_BULLETS else [
                    "Synthetic fermentation dataset from the IndPenSim benchmark.",
                    "Signals from a 100 kL bioreactor sampled every 12 minutes.",
                    "500 simulated batches with multi-sensor time series.",
                ],
            ),
            NarrativeSection(
                key="data-provenance",
                title="Data provenance",
                bullets=[
                    "Source: IndPenSim simulation of industrial penicillin fermentation ([Goldrick et al., 2019](https://www.sciencedirect.com/science/article/pii/S0098135418305106)).",
                    "Lineage: notebook data-indpensim.ipynb generates the snapshot.",
                    "Versioning: MLflow run tags `train_commit` and `test_commit` record the code used for training and testing.",
                ],
            ),
            NarrativeSection(
                key="data-quality",
                title="Data quality and splits",
                bullets=_DATA_QUALITY_DETAILS + [
                    "Train, validation, and test splits are defined at the batch level (70/15/15).",
                ] if _DATA_QUALITY_DETAILS else [
                    "Simulation data include controlled noise and no sensor drift.",
                    "Offline target measurements are mostly missing and dropped before training.",
                    "Train, validation, and test splits are defined at the batch level (70/15/15).",
                ],
            ),
            NarrativeSection(
                key="odd-definition",
                title="ODD definition",
                body=(
                    "Normal penicillin fermentation conditions in a simulated 100 kL reactor. "
                    "A data-driven ODD boundary is defined by KDE on the train split per feature or feature pair. "
                    "We select the highest-density region that covers approximately 98% probability mass (after optional quantile clipping). "
                    "This region is shown as a one-dimensional density band or a two-dimensional superlevel-set hull."
                ),
            ),
            NarrativeSection(
                key="scenario-catalog",
                title="Scenario catalog",
                body=(
                    "Stress scenarios perturb input signals to mimic realistic sensor and process faults with controlled severities."
                ),
                bullets=[
                    "Scenario types include sensor noise, missing channels, stuck sensors, spikes, timing offsets, wrong-state episodes, and chattering faults ([Windmann et al., 2025](https://ieeexplore.ieee.org/abstract/document/11205527)).",
                ],
            ),
            NarrativeSection(
                key="distribution-diagnostics",
                title="Distribution diagnostics",
                # body=(
                # ),
                bullets=[
                    "The ODD region is defined as the 98%-mass KDE of the train data.",
                    "The highest train-to-test drift appears for the feature 'dissolved_oxygen' in the drift bar plot.",
                    "The KDE overlays for 'dissolved_oxygen' show that the drift remains within the ODD region, even in the 'Drift' test scenario.",
                    "The 'dissolved_oxygen' feature should be monitored in live deployments to detect potential drift beyond the ODD.",
                ],
            ),
        ],
    ),
    "evaluation": NarrativeBlock(
        sections=[
            NarrativeSection(
                key="performance-kpis",
                title="Performance KPIs",
                body="This section summarizes baseline performance and acceptance thresholds for the selected model.",
                bullets=[
                    "Baseline accuracy uses MSE and MAPE on validation and test data with a purge window applied, and results are reported alongside perturbed KPIs.",
                    "Acceptance thresholds are test MSE ≤ 0.04, test MAPE ≤ 1.0, robustness score (severity–performance curve via R_mean) ≥ 0.90.",
                ],
            ),
            NarrativeSection(
                key="uncertainty-quantification",
                title="Uncertainty quantification",
                body="Quantile regression is evaluated with the weighted interval score (WIS) on test and UQ slices.",
                bullets=[
                    "WIS summarizes calibration and sharpness for prediction intervals. Lower values are better.",
                ],
            ),
            NarrativeSection(
                key="robustness-evidence",
                title="Robustness evidence",
                # body="Scenario radar shows robustness scores per perturbation, robustness summary compares perturbed vs clean baselines.",
                bullets=[
                    # "Scenarios and mean robustness score from ([Windmann et al., 2025](https://ieeexplore.ieee.org/abstract/document/11205527)).",
                    "Latest model version improves MSE on the clean validation and test sets.",
                    "However, MAPE and the robustness score in the 'Noise' scenario have decreased. Overfitting?",
                ],
            ),
        ],
    ),
    "limitations": NarrativeBlock(
        body=(
            "The dataset is synthetic and the stress scenarios are finite, so results may miss real fermentation variability and plant-specific instrumentation quirks."
        ),
        bullets=[
            "Penicillin targets are validated offline only, no closed-loop deployment has been attempted.",
            "Scenario catalog omits extended fouling periods, rare mechanical faults, and upstream feed excursions.",
            # "Expect data drift in live plants with different sensors or cleaning cycles, so continuous monitoring is required.",
            "To fulfill the EU AI Act requirements, further documentation on data governance and risk management is needed.",
        ],
        # footer_note="Next steps: extend stress library, add live telemetry back-testing, secure governance sign-off.",
    ),
}
