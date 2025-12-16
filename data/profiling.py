from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Any
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DatasetProfile:
    summary: pd.DataFrame
    histograms: Dict[str, Dict[str, Any]]
    kde_samples: Dict[str, Dict[str, List[float]]]
    scenario_kde_samples: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


def profile_to_payload(profile: DatasetProfile, *, dataset_key: str) -> Dict[str, Any]:
    return {
        "dataset": dataset_key,
        "summary_columns": list(profile.summary.columns),
        "summary_rows": profile.summary.to_dict(orient="records"),
        "histograms": profile.histograms,
        "kde_samples": profile.kde_samples,
        "scenario_kde_samples": profile.scenario_kde_samples,
    }


def payload_to_profile(payload: Dict[str, Any]) -> DatasetProfile:
    columns = payload.get("summary_columns")
    rows = payload.get("summary_rows", [])
    summary_df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
    histograms = payload.get("histograms", {})
    kde_samples = payload.get("kde_samples", {})
    scenario_kde_samples = payload.get("scenario_kde_samples", {})
    return DatasetProfile(
        summary=summary_df,
        histograms=histograms,
        kde_samples=kde_samples,
        scenario_kde_samples=scenario_kde_samples,
    )


def dump_profile_payload(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_profile_from_path(path: str) -> DatasetProfile:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload_to_profile(payload)


def _format_numeric_summary(values: pd.Series) -> str:
    if values.empty:
        return "n/a"
    mean = values.mean()
    std = values.std()
    q05 = values.quantile(0.05)
    q95 = values.quantile(0.95)
    return f"μ={mean:.3f}, σ={std:.3f}, q05={q05:.3f}, q95={q95:.3f}"


def compute_dataset_profile_from_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_columns: Sequence[str],
    *,
    target_columns: Sequence[str] | None = None,
    sample_cap: int = 20000,
    seed: int | None = 42,
) -> DatasetProfile:
    rng = np.random.default_rng(seed)
    cols = list(dict.fromkeys(list(input_columns) + (list(target_columns) if target_columns else [])))
    train_df = train_df.loc[:, cols].copy()
    val_df = val_df.loc[:, cols].copy()
    test_df = test_df.loc[:, cols].copy()

    # Use full frames (no sampling)

    records: List[Dict[str, object]] = []
    histogram_specs: Dict[str, Dict[str, List[float] | int]] = {}
    for column in train_df.columns:
        train_series = train_df[column].dropna()
        test_series = test_df[column].dropna()
        dtype = "numeric"
        val_series = val_df[column].dropna()
        score_metric = "KS"
        score_train = ks_2samp(train_series.to_numpy(dtype=float), test_series.to_numpy(dtype=float)).statistic
        score_val = ks_2samp(val_series.to_numpy(dtype=float), test_series.to_numpy(dtype=float)).statistic
        
        summary_train = _format_numeric_summary(train_series)
        summary_val = _format_numeric_summary(val_series)
        summary_test = _format_numeric_summary(test_series)

        combined = pd.concat([train_series, test_series], ignore_index=True)
        if combined.nunique() >= 2:
            bin_count = min(40, max(10, int(np.sqrt(len(combined)))))
            edges = np.histogram_bin_edges(combined.to_numpy(dtype=float), bins=bin_count)
        else:
            unique_val = combined.iloc[0] if not combined.empty else 0.0
            edges = np.linspace(unique_val - 0.5, unique_val + 0.5, num=3)
        train_hist, _ = np.histogram(train_series.to_numpy(dtype=float), bins=edges, density=True)
        test_hist, _ = np.histogram(test_series.to_numpy(dtype=float), bins=edges, density=True)
        histogram_specs[column] = {
            "bins": edges.tolist(),
            "train_density": train_hist.tolist(),
            "test_density": test_hist.tolist(),
            "train_count": int(len(train_series)),
            "test_count": int(len(test_series)),
        }

        records.append(
            {
                "feature": column,
                "dtype": dtype,
                "score_metric": score_metric,
                "score_train_test": float(score_train),
                "score_val_test": float(score_val),
                "train_summary": summary_train,
                "val_summary": summary_val,
                "test_summary": summary_test,
                "train_count": int(len(train_series)),
                "val_count": int(len(val_series)),
                "test_count": int(len(test_series)),
            }
        )

    profile_df = pd.DataFrame(records)
    profile_df.sort_values("score_train_test", ascending=False, inplace=True)
    profile_df.reset_index(drop=True, inplace=True)

    def _sample_values(values: pd.Series, cap: int) -> List[float]:
        arr = values.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > cap:
            idx = rng.choice(arr.size, size=cap, replace=False)
            arr = arr[idx]
        return arr.tolist()

    columns = list(train_df.columns)
    kde_samples: Dict[str, Dict[str, List[float]]] = {}
    for split_name, frame in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        sampled = frame.loc[:, columns]
        kde_samples[split_name] = {
            col: _sample_values(sampled[col], sample_cap)
            for col in columns
        }

    return DatasetProfile(
        summary=profile_df,
        histograms=histogram_specs,
        kde_samples=kde_samples,
        scenario_kde_samples={},
    )
