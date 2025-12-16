from __future__ import annotations

import pandas as pd


def _to_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def metric_summary_rows(baseline_row):
    if baseline_row is None:
        return []
    metric_summary = baseline_row.get("metric_summary")
    if not isinstance(metric_summary, dict) or not metric_summary:
        return []
    primary_metric = baseline_row.get("primary_metric")
    ordered = []
    if isinstance(primary_metric, str) and primary_metric in metric_summary:
        ordered.append(primary_metric)
    for name in sorted(metric_summary.keys(), key=lambda value: value.lower()):
        if name not in ordered:
            ordered.append(name)
    rows = []
    for metric_name in ordered:
        splits = metric_summary.get(metric_name) or {}
        val = _to_float(splits.get("val"))
        test = _to_float(splits.get("test"))
        pert = _to_float(splits.get("pert"))
        rows.append((metric_name, (val, test, pert)))
    return rows


def quantile_wis_entry(baseline_row):
    if baseline_row is None:
        return None
    clean = _to_float(baseline_row.get("quantile_wis_clean"))
    pert = _to_float(baseline_row.get("quantile_wis_pert"))
    if clean is None and pert is None:
        return None
    label = baseline_row.get("quantile_wis_label") or "Quantile WIS"
    return label, clean, pert


def r_mean_value(baseline_row):
    if baseline_row is None:
        return None
    return _to_float(baseline_row.get("R_mean"))
