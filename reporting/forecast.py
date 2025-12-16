from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd


@dataclass
class ForecastSample:
    sample_id: int
    input_time: np.ndarray
    output_time: np.ndarray
    clean_input: np.ndarray
    perturbed_input: np.ndarray | None
    target: np.ndarray
    prediction: np.ndarray
    quantile_predictions: Dict[float, np.ndarray]
    scenario_label: str
    severity: float
    input_feature_names: List[str]
    target_feature_names: List[str]
    affected_feature_names: List[str]


def download_forecast_payload(
    client: mlflow.MlflowClient,
    run_id: str,
    *,
    phi_tag: str | None,
    test_metric: str | None,
) -> dict:
    if not run_id:
        raise ValueError("Run ID required to download forecast payload.")
    if not phi_tag:
        raise ValueError(f"Missing phi_tag for run {run_id}.")
    metric = test_metric or "MSE"
    artifact_path = f"robustness/{phi_tag}/{metric}/forecast_samples.json"
    local_path = client.download_artifacts(str(run_id), artifact_path)
    with open(local_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _download_forecast_payload(
    client: mlflow.MlflowClient,
    row: pd.Series,
    *,
    default_phi: str | None = None,
    default_metric: str | None = None,
) -> dict:
    run_id = row.get("run_id")
    if not run_id:
        raise ValueError("Row is missing run_id, cannot download forecast artifact.")
    phi_tag = row.get("phi_tag") or default_phi
    test_metric = row.get("test_metric") or default_metric or "MSE"
    return download_forecast_payload(
        client,
        str(run_id),
        phi_tag=phi_tag,
        test_metric=test_metric,
    )


def _load_quantile_predictions(
    client: mlflow.MlflowClient,
    quantile_rows: pd.DataFrame | None,
    baseline_row: pd.Series,
) -> Dict[int, Dict[float, np.ndarray]]:
    if quantile_rows is None or quantile_rows.empty:
        return {}

    default_phi = baseline_row.get("phi_tag")
    default_metric = baseline_row.get("test_metric")
    predictions: Dict[int, Dict[float, np.ndarray]] = {}
    for _, row in quantile_rows.iterrows():
        payload = _download_forecast_payload(
            client,
            row,
            default_phi=default_phi,
            default_metric=default_metric,
        )
        level = payload.get("quantile_level") or row.get("quantile_level")
        if level is None:
            raise ValueError(f"Quantile run {row.get('run_id')} is missing quantile_level metadata.")
        level = float(level)
        for entry in payload.get("samples", []):
            sample_id_raw = entry.get("sample_id")
            if sample_id_raw is None:
                continue
            sample_id = int(sample_id_raw)
            predictions.setdefault(sample_id, {})[level] = np.asarray(entry["prediction"], dtype=float)
    return predictions


def load_forecast_samples(
    client: mlflow.MlflowClient,
    baseline_row: pd.Series,
    quantile_rows: pd.DataFrame | None,
) -> List[ForecastSample]:
    payload = _download_forecast_payload(client, baseline_row)
    baseline_samples = payload.get("samples") or []
    if not baseline_samples:
        raise ValueError("No forecast samples logged for baseline run.")

    input_feature_names = payload.get("input_feature_names") or baseline_samples[0].get("input_feature_names")
    target_feature_names = payload.get("target_feature_names") or baseline_samples[0].get("target_feature_names")
    if not input_feature_names or not target_feature_names:
        raise ValueError(
            "Forecast artifact missing feature names. Please rerun run_testing to regenerate artifacts with metadata."
        )

    quantile_predictions = _load_quantile_predictions(client, quantile_rows, baseline_row)
    samples: List[ForecastSample] = []
    for entry in baseline_samples:
        sample_id = int(entry["sample_id"])
        input_matrix = np.asarray(entry["input"], dtype=float)
        target_matrix = np.asarray(entry["target"], dtype=float)
        prediction_matrix = np.asarray(entry["prediction"], dtype=float)

        input_time_index = np.arange(1, input_matrix.shape[0] + 1, dtype=float)
        output_time_index = np.arange(input_matrix.shape[0] + 1, input_matrix.shape[0] + target_matrix.shape[0] + 1, dtype=float)

        perturbed_input = None
        if entry.get("input_perturbed") is not None:
            perturbed_input = np.asarray(entry["input_perturbed"], dtype=float)

        scenario_label_raw = entry.get("perturbation_name")
        if not scenario_label_raw:
            raise ValueError("Forecast artifact missing perturbation name.")
        scenario_label = str(scenario_label_raw).replace("_", " ").title()
        severity = float(entry.get("severity", float("nan")))

        affected_names = entry.get("affected_channel_names") or []

        samples.append(
            ForecastSample(
                sample_id=sample_id,
                input_time=input_time_index,
                output_time=output_time_index,
                clean_input=input_matrix,
                perturbed_input=perturbed_input,
                target=target_matrix,
                prediction=prediction_matrix,
                quantile_predictions=quantile_predictions.get(sample_id, {}),
                scenario_label=scenario_label,
                severity=severity,
                input_feature_names=list(input_feature_names),
                target_feature_names=list(target_feature_names),
                affected_feature_names=list(affected_names),
            )
        )
    return samples
