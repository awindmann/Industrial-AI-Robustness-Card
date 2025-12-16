from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mlflow
from mlflow.entities import Experiment, Run


@dataclass(frozen=True)
class RunRecord:
    run: Run
    experiment: Experiment

    @property
    def run_id(self) -> str:
        return self.run.info.run_id

    @property
    def dataset(self) -> Optional[str]:
        return self.run.data.tags.get("dataset")

    @property
    def model_architecture(self) -> Optional[str]:
        return self.run.data.tags.get("model_architecture")

    @property
    def stage(self) -> Optional[str]:
        value = self.run.data.tags.get("stage")
        return value.lower() if value else None


def get_mlflow_client(tracking_uri: str) -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.MlflowClient()


def find_experiments(
    client: mlflow.MlflowClient,
    prefix: str,
) -> list[Experiment]:
    experiments = client.search_experiments()
    return [exp for exp in experiments if exp.name.startswith(prefix)]


def collect_runs(
    client: mlflow.MlflowClient,
    experiments: Iterable[Experiment],
    max_results: int = 10000,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for experiment in experiments:
        runs = client.search_runs([experiment.experiment_id], max_results=max_results)
        for run in runs:
            records.append(RunRecord(run=run, experiment=experiment))
    return records
