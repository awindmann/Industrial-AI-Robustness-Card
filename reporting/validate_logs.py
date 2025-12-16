from __future__ import annotations

import argparse
from pathlib import Path

from data.datasets import resolve_with_defaults
from reporting.data_access import collect_runs, find_experiments, get_mlflow_client


def _build_tracking_uri(logdir: str) -> str:
    if str(logdir).startswith("http://"):
        return logdir
    path = Path(logdir)
    uri_path = path.as_posix()
    return f"file:{uri_path if path.is_absolute() else f'./{uri_path}'}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _dataset_keys(dataset_args: list[str] | None) -> list[str]:
    specs = resolve_with_defaults(
        default_datasets=dataset_args,
        default_targets=None,
        datasets=dataset_args,
        targets=None,
    )
    return [spec.key for spec in specs]


def _verify_backbone(run, dataset: str) -> None:
    params = run.data.params
    tags = run.data.tags
    run_id = run.info.run_id
    phi_tag = params.get("phi_tag")
    test_metric = params.get("test_metric")

    _require(params.get("tested") == "true", f"Run {run_id} on {dataset} is not marked tested=true")
    _require(phi_tag, f"Run {run_id} on {dataset} missing phi_tag")
    _require(test_metric, f"Run {run_id} on {dataset} missing test_metric")

    metric_prefix = f"{phi_tag}/{test_metric}"
    required_metrics = [
        f"{test_metric}_val",
        f"{test_metric}_test",
        f"{metric_prefix}/R_mean",
    ]
    for key in required_metrics:
        _require(key in run.data.metrics, f"Run {run_id} on {dataset} missing metric {key}")


def _verify_quantile(run, dataset: str) -> None:
    tags = run.data.tags
    params = run.data.params
    run_id = run.info.run_id
    _require(params.get("tested") == "true", f"Quantile run {run_id} on {dataset} is not marked tested=true")
    _require(tags.get("linked_backbone_run"), f"Quantile run {run_id} on {dataset} missing linked_backbone_run")
    _require(tags.get("quantile_level") is not None, f"Quantile run {run_id} on {dataset} missing quantile_level")
    _require(params.get("phi_tag"), f"Quantile run {run_id} on {dataset} missing phi_tag")
    _require(params.get("test_metric"), f"Quantile run {run_id} on {dataset} missing test_metric")


def _require_artifacts(client, run, phi_tag: str, test_metric: str) -> None:
    client.download_artifacts(run.info.run_id, "dataset_stats/dataset_profile.json")
    client.download_artifacts(run.info.run_id, f"robustness/{phi_tag}/{test_metric}/forecast_samples.json")


def validate_logs(logdir: str, experiment_prefix: str, datasets: list[str] | None) -> None:
    tracking_uri = _build_tracking_uri(logdir)
    client = get_mlflow_client(tracking_uri)
    experiments = find_experiments(client, experiment_prefix)
    _require(len(experiments) > 0, f"No experiments found with prefix {experiment_prefix}")

    dataset_filter = set(_dataset_keys(datasets)) if datasets else None
    records = [
        rec
        for rec in collect_runs(client, experiments)
        if getattr(rec.run.info, "lifecycle_stage", "active") == "active"
    ]
    _require(len(records) > 0, "No runs discovered in the selected experiments")

    runs_by_dataset: dict[str, list] = {}
    for rec in records:
        ds = rec.dataset
        _require(ds is not None, f"Run {rec.run_id} missing dataset tag")
        if dataset_filter and ds not in dataset_filter:
            continue
        runs_by_dataset.setdefault(ds, []).append(rec)

    _require(len(runs_by_dataset) > 0, "No runs matched the requested datasets")

    for dataset, run_records in runs_by_dataset.items():
        backbones = [r for r in run_records if (r.stage == "train")]
        quantiles = [r for r in run_records if (r.stage == "uq")]
        _require(len(backbones) > 0, f"Dataset {dataset} has no backbone runs")

        for rec in backbones:
            _verify_backbone(rec.run, dataset)
            phi_tag = rec.run.data.params["phi_tag"]
            test_metric = rec.run.data.params["test_metric"]
            _require_artifacts(client, rec.run, phi_tag, test_metric)

        for rec in quantiles:
            _verify_quantile(rec.run, dataset)
            phi_tag = rec.run.data.params["phi_tag"]
            test_metric = rec.run.data.params["test_metric"]
            _require_artifacts(client, rec.run, phi_tag, test_metric)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="MLflow tracking directory or HTTP URI")
    parser.add_argument("--experiment-prefix", required=True, help="Experiment prefix to validate")
    parser.add_argument("--dataset", nargs="*", default=None, help="Datasets to validate (defaults to all in registry)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validate_logs(args.logdir, args.experiment_prefix, args.dataset)
