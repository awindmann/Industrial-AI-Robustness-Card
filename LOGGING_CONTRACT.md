# MLflow Logging Contract for Reporting

Reporting assumes a local MLflow file store under `logs/` (or any tracking URI passed to `--logdir`). Experiments must be named `{MLFLOW_EXPERIMENT_PREFIX}-{dataset}`.

## Required runs
- Backbone runs (stage=`train`) with `tested=true`
- Quantile runs (stage=`uq`) linked to a backbone via `linked_backbone_run` and tagged with `quantile_level`

## Required tags/params
- `dataset`
- `stage` ∈ {`train`, `uq`}
- `model_architecture`
- `tested` = `true` (backbone and quantile runs that should surface in the report)
- `phi_tag` (e.g., `pow_r3_phi0.2`)
- `test_metric` (e.g., `MAPE`)
- Quantile runs: `linked_backbone_run`, `quantile_level`

## Required metrics (per backbone)
- `<test_metric>_val`
- `<test_metric>_test`
- Robustness metrics under `{phi_tag}/{test_metric}/…`:
  - `R_mean`
  - `scenario/<ScenarioName>/R_mean`
- Optional (used when present): `quantile/WIS_<lower>_<upper>_clean`, `quantile/WIS_<lower>_<upper>_pert`

## Required artifacts
- `dataset_stats/dataset_profile.json` (payload from `data.profiling.profile_to_payload`)
- `robustness/{phi_tag}/{test_metric}/forecast_samples.json` on backbone runs
- Quantile runs: `robustness/{phi_tag}/{test_metric}/forecast_samples.json`

If any element above is missing, `reporting.validate_logs` exits with an error. The bundled example uses relative `file:./logs/...` artifact URIs, run commands from the repository root.
