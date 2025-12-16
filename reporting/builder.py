from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Sequence
import pandas as pd

from reporting.data_access import RunRecord
from data.profiling import load_profile_from_path


LOSS_KEYS = ("loss_fn", "loss_function", "criterion", "loss", "objective", "test_metric")
METRIC_SUFFIX_MAP = {
    "_val": "val",
    "_test": "test",
    "_pert": "pert",
}


def _match_metric_name(candidates: Iterable[str], target: str | None) -> Optional[str]:
    if not target:
        return None
    target_key = str(target).strip().lower()
    if not target_key:
        return None
    for name in candidates:
        if str(name).strip().lower() == target_key:
            return name
    return None


def _sorted_metric_names(names: Iterable[str]) -> List[str]:
    return sorted({str(name) for name in names if name}, key=lambda value: value.lower())


def _parse_quantile_level_tag(tag_value: Optional[str]) -> List[float]:
    if not tag_value:
        return []
    levels: List[float] = []
    for token in str(tag_value).split(","):
        token = token.strip()
        if not token:
            continue
        levels.append(float(token))
    return sorted(levels)


def _collect_metric_summary(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        for suffix, label in METRIC_SUFFIX_MAP.items():
            if key.endswith(suffix):
                metric_name = key[: -len(suffix)]
                if not metric_name:
                    continue
                if metric_name.startswith("WIS_") or metric_name.startswith("quantile/WIS_"):
                    continue
                bucket = summary.setdefault(metric_name, {})
                bucket[label] = float(value)
                break
    return summary


def _resolve_primary_metric(
    metric_summary: Dict[str, Dict[str, float]],
    params: Dict[str, str],
    tags: Dict[str, str],
) -> Optional[str]:
    if not metric_summary:
        return None
    metric_names = list(metric_summary.keys())
    loss_candidates = []
    for key in LOSS_KEYS:
        candidate = params.get(key) or tags.get(key)
        if candidate:
            loss_candidates.append(candidate)
    for candidate in loss_candidates:
        match = _match_metric_name(metric_names, candidate)
        if match:
            return match
    sorted_names = _sorted_metric_names(metric_names)
    return sorted_names[0] if sorted_names else None


def _resolve_robust_metric_name(
    metrics: Dict[str, float],
    phi_tag: str | None,
    preferred_metric: str | None,
) -> Optional[str]:
    if not phi_tag:
        return preferred_metric
    prefix = f"{phi_tag}/"
    available: Dict[str, str] = {}
    for key in metrics:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        if "/" not in remainder:
            continue
        metric_token = remainder.split("/", 1)[0]
        if metric_token:
            available.setdefault(metric_token.lower(), metric_token)
    if not available:
        return preferred_metric
    if preferred_metric:
        match = available.get(str(preferred_metric).strip().lower())
        if match:
            return match
    return _sorted_metric_names(available.values())[0]


def _score_run(run: RunRecord) -> float:
    metrics = run.run.data.metrics
    params = run.run.data.params
    test_metric = params.get("test_metric")
    priority = ["best_val_loss"]
    if test_metric:
        priority.append(f"{test_metric}_val")
    for key in priority:
        value = metrics.get(key)
        if value is not None:
            return value
    return float("inf")


def _extract_metrics(run: RunRecord) -> Dict[str, object]:
    metrics = run.run.data.metrics
    params = run.run.data.params
    tags = run.run.data.tags

    run_test_method = params.get("test_method")
    phi_tag = params.get("phi_tag")

    metric_summary = _collect_metric_summary(metrics)
    primary_metric = _resolve_primary_metric(metric_summary, params, tags)
    robustness_metric = _resolve_robust_metric_name(metrics, phi_tag, primary_metric)

    result: Dict[str, object] = {
        "metric_summary": metric_summary,
        "primary_metric": primary_metric,
    }

    stage_name = (run.run.data.tags.get("stage") or "train").lower()
    if stage_name == "train":
        quantile_levels = _parse_quantile_level_tag(tags.get("uq/quantile_levels"))
        if len(quantile_levels) >= 2:
            lower_level = quantile_levels[0]
            upper_level = quantile_levels[-1]
            label = f"WIS {lower_level:g}-{upper_level:g}"
            clean_key = f"quantile/WIS_{lower_level:g}_{upper_level:g}_clean"
            pert_key = f"quantile/WIS_{lower_level:g}_{upper_level:g}_pert"
            result["quantile_wis_label"] = label
            result["quantile_wis_clean"] = metrics[clean_key]
            result["quantile_wis_pert"] = metrics[pert_key]

    preferred_suffix = "_canonical" if run_test_method == "brute_force" else ""

    def _get_metric(name: str):
        if not phi_tag or not robustness_metric:
            return None
        base_key = f"{phi_tag}/{robustness_metric}"
        key = f"{base_key}/{name}{preferred_suffix}"
        value = metrics.get(key)
        if value is not None:
            return value
        alt_suffix = "" if preferred_suffix == "_canonical" else "_canonical"
        alt_key = f"{base_key}/{name}{alt_suffix}"
        return metrics.get(alt_key)

    result["R_mean"] = _get_metric("R_mean")

    scenario_prefix = (
        f"{phi_tag}/{robustness_metric}/scenario/"
        if phi_tag and robustness_metric
        else None
    )
    if scenario_prefix:
        scenario_mean_candidates = {}
        for metric_key, metric_value in metrics.items():
            if not metric_key.startswith(scenario_prefix):
                continue
            remainder = metric_key[len(scenario_prefix):]
            if "/" not in remainder:
                continue
            scenario_name, metric_label = remainder.split("/", 1)
            suffix = "_canonical" if metric_label.endswith("_canonical") else ""
            if suffix:
                metric_label = metric_label[: -len("_canonical")]
            if metric_label != "R_mean":
                continue
            bucket = scenario_mean_candidates.setdefault(scenario_name, {})
            bucket[suffix] = metric_value

        if not scenario_mean_candidates:
            raise KeyError(
                f"Run {run.run.info.run_id} logged no scenario R_mean metrics under {scenario_prefix}."
            )
        for scenario_name, options in scenario_mean_candidates.items():
            if preferred_suffix not in options:
                raise KeyError(
                    f"Run {run.run.info.run_id} missing scenario R_mean metric for '{scenario_name}' "
                    f"with suffix '{preferred_suffix}'. Available: {list(options.keys())}"
                )
            result[f"scenario_{scenario_name}"] = options[preferred_suffix]

    return result


def _summarize_run(run: RunRecord) -> Dict[str, object]:
    tags = run.run.data.tags
    params = run.run.data.params

    summary: Dict[str, object] = {
        "run_id": run.run.info.run_id,
        "experiment": run.experiment.name,
        "dataset": tags.get("dataset"),
        "stage": (tags.get("stage") or "train").lower(),
        "model": tags.get("model_architecture"),
        "model_loader": tags.get("model_loader"),
        "backbone_architecture": tags.get("backbone_architecture") or tags.get("model_architecture"),
        "test_method": params.get("test_method"),
        "test_metric": params.get("test_metric"),
        "phi": params.get("phi"),
        "phi_tag": params.get("phi_tag"),
        "linked_backbone_run": tags.get("linked_backbone_run"),
        "quantile_level": tags.get("quantile_level"),
        "input_len": params.get("input_len"),
        "target_len": params.get("target_len"),
        "tested": params.get("tested"),
        "forecast_sample_count": params.get("forecast_sample_count"),
        "best_model_tag": tags.get("best_model"),
        "train_commit": tags.get("train_commit"),
        "test_commit": tags.get("test_commit"),
    }
    summary.update(_extract_metrics(run))
    return summary


@dataclass
class RunReport:
    record: RunRecord
    tables: Dict[str, pd.DataFrame]


@dataclass
class ReportingBundle:
    tracking_uri: str
    runs: Dict[str, RunReport]
    odd_kde_mass: float = 0.98


def _matches_model(name: Optional[str], model_filter: Optional[str]) -> bool:
    if model_filter is None:
        return True
    if name is None:
        return False
    return name.lower() == model_filter.lower()


def _build_dataset_report(
    records: List[RunRecord],
    client,
    model_filter: str | None = None,
) -> Tuple[List[RunRecord], Dict[str, pd.DataFrame]]:
    if not records:
        raise ValueError("No records provided for dataset report.")

    baseline_candidates: Dict[str, List[Tuple[float, RunRecord]]] = {}
    quantile_by_parent: Dict[str, Dict[float, RunRecord]] = {}

    def _is_tested(record: RunRecord) -> bool:
        return str(record.run.data.params.get("tested")).lower() == "true"

    for record in records:
        if getattr(record.run.info, "lifecycle_stage", "active") != "active":
            continue
        tags = record.run.data.tags
        stage = (tags.get("stage") or "train").lower()
        score = _score_run(record)
        if stage == "train":
            architecture = tags.get("model_architecture")
            if architecture and _matches_model(architecture, model_filter):
                baseline_candidates.setdefault(architecture, []).append((score, record))
        elif stage == "uq":
            parent = tags.get("linked_backbone_run")
            level = tags.get("quantile_level")
            if parent and level is not None:
                level_value = float(level)
                quantile_by_parent.setdefault(parent, {})[level_value] = record

    baseline_records = []
    best_baseline_ids = {}
    summary_cache = {}

    def _summary(rec: RunRecord) -> Dict[str, object]:
        rid = rec.run.info.run_id
        if rid not in summary_cache:
            summary_cache[rid] = _summarize_run(rec)
        return summary_cache[rid]
    for architecture, candidates in baseline_candidates.items():
        tested_candidates = [
            (score, record) for score, record in candidates if _is_tested(record)
        ]
        if not tested_candidates:
            raise ValueError(
                f"No tested baseline runs found for architecture '{architecture}'."
            )
        sorted_candidates = sorted(tested_candidates, key=lambda item: item[0])
        best_baseline_ids[architecture] = sorted_candidates[0][1].run.info.run_id
        baseline_records.extend(record for _, record in sorted_candidates)


    if not baseline_records:
        raise ValueError("No tested baseline runs found.")

    baseline_ids = {record.run.info.run_id for record in baseline_records}
    filtered_quantiles = {}
    selected_quantiles = []
    for parent_id, entries in quantile_by_parent.items():
        if baseline_ids and parent_id not in baseline_ids:
            continue
        valid_entries = {
            level: record for level, record in entries.items() if _is_tested(record)
        }
        if not valid_entries:
            continue
        filtered_quantiles[parent_id] = valid_entries
        selected_quantiles.extend(valid_entries.values())

    quantile_by_parent = filtered_quantiles

    selected_records: List[RunRecord] = baseline_records + selected_quantiles
    best_architecture_ids = set(best_baseline_ids.values())
    if not selected_records:
        raise ValueError("No evaluated runs found for dataset.")

    rows = []
    for record in selected_records:
        summary = _summary(record)
        parent = summary.get("linked_backbone_run")
        if summary.get("stage") == "train":
            summary["is_architecture_best"] = summary["run_id"] in best_architecture_ids
        else:
            summary["is_architecture_best"] = False
        if summary.get("stage") == "train" and summary.get("run_id") in quantile_by_parent:
            levels = sorted(quantile_by_parent[summary["run_id"]].keys())
            summary["quantile_levels_available"] = ",".join(f"{lvl:g}" for lvl in levels)
            summary["quantile_run_ids"] = ",".join(
                f"{lvl:g}:{quantile_by_parent[summary['run_id']][lvl].run.info.run_id}"
                for lvl in levels
            )
        elif parent and parent in quantile_by_parent:
            levels = sorted(quantile_by_parent[parent].keys())
            summary["quantile_levels_available"] = ",".join(f"{lvl:g}" for lvl in levels)
            summary["quantile_run_ids"] = ",".join(
                f"{lvl:g}:{quantile_by_parent[parent][lvl].run.info.run_id}"
                for lvl in levels
            )
        rows.append(summary)

    full_results_df = pd.DataFrame(rows)

    best_baseline = min(
        (rec for rec in selected_records if (_summary(rec)["stage"] == "train")),
        key=_score_run,
        default=selected_records[0],
    )

    tables = {
        "full_results": full_results_df,
    }
    if client is None:
        raise ValueError("MLflow client required to load dataset profile artifacts. Dataset profile artifact is produced by run_testing.py.")
    artifact_path = client.download_artifacts(
        best_baseline.run.info.run_id,
        "dataset_stats/dataset_profile.json",
    )
    profile = load_profile_from_path(artifact_path)
    tables["data_profile"] = profile.summary
    tables["data_histograms"] = profile.histograms
    data_kde = dict(profile.kde_samples)
    if profile.scenario_kde_samples:
        data_kde.update(profile.scenario_kde_samples)
    tables["data_kde"] = data_kde
    return baseline_records, tables


def build_reporting_bundle(
    tracking_uri: str,
    run_records: Iterable[RunRecord],
    mlflow_client,
    dataset_filters: Sequence[str] | None = None,
    model_filter: str | None = None,
    odd_kde_mass: float = 0.98,
) -> ReportingBundle:
    odd_kde_mass = float(odd_kde_mass)
    if not (0.0 < odd_kde_mass < 1.0):
        raise ValueError(f"Invalid odd_kde_mass {odd_kde_mass}; must be in (0,1).")
    run_records = list(run_records)
    runs = {}
    grouped = {}
    dataset_filter_set = {str(item) for item in dataset_filters} if dataset_filters else None
    for record in run_records:
        dataset = record.run.data.tags.get("dataset")
        if not dataset:
            raise ValueError(f"Run {record.run.info.run_id} is missing the 'dataset' tag.")
        if dataset_filter_set is not None and dataset not in dataset_filter_set:
            continue
        grouped.setdefault(dataset, []).append(record)

    for dataset, records in grouped.items():
        baseline_records, tables = _build_dataset_report(records, mlflow_client, model_filter=model_filter)
        for baseline_record in baseline_records:
            runs[baseline_record.run.info.run_id] = RunReport(record=baseline_record, tables=tables)

    return ReportingBundle(
        tracking_uri=tracking_uri,
        runs=runs,
        odd_kde_mass=odd_kde_mass,
    )
