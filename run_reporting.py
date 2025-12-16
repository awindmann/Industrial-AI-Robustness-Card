import argparse
from pathlib import Path

import config as config
from data.datasets import resolve_with_defaults
from reporting.builder import build_reporting_bundle
from reporting.data_access import (
    collect_runs,
    find_experiments,
    get_mlflow_client,
)
from reporting.interactive_app import launch_dash_app


def build_tracking_uri(logdir: str) -> str:
    if logdir.startswith("http://"):
        return logdir
    path = Path(logdir)
    uri_path = path.as_posix()
    return f"file:{uri_path if path.is_absolute() else f'./{uri_path}'}"


def run_reporting(args):
    tracking_uri = build_tracking_uri(args.logdir)
    client = get_mlflow_client(tracking_uri)
    odd_kde_mass = float(getattr(args, "odd_kde_mass", 0.98))
    if not (0.0 < odd_kde_mass < 1.0):
        raise ValueError(f"Invalid odd_kde_mass {odd_kde_mass}; must be in (0,1).")

    experiment_prefix = args.mlflow_experiment_prefix
    experiments = find_experiments(client, experiment_prefix)
    if not experiments:
        print(f"No experiments found with prefix {experiment_prefix}.")
        return

    resolved_specs = resolve_with_defaults(
        default_datasets=args.data_files,
        default_targets=args.data_targets,
        datasets=getattr(args, "data_files", None),
        targets=getattr(args, "data_targets", None),
    )

    print(f"Tracking URI: {tracking_uri}")
    print(f"Resolved datasets: {[spec.key for spec in resolved_specs]}")

    run_records = collect_runs(client, experiments)
    print(f"Discovered {len(run_records)} runs across {len(experiments)} experiments.")

    dataset_filters = [spec.key for spec in resolved_specs] if resolved_specs else None
    model_filter = None
    model_arg = getattr(args, "model", None)
    if isinstance(model_arg, str) and model_arg.lower() != "all":
        model_filter = model_arg

    bundle = build_reporting_bundle(
        tracking_uri=tracking_uri,
        run_records=run_records,
        mlflow_client=client,
        dataset_filters=dataset_filters,
        model_filter=model_filter,
        odd_kde_mass=odd_kde_mass,
    )
    print(f"Reporting bundle contains {len(bundle.runs)} reports.")

    if not bundle.runs:
        print("No reporting tables found in the discovered runs.")
        return

    if args.report_serve:
        print(f"Serving interactive dashboard on port {args.report_serve_port}.")
        launch_dash_app(bundle, port=args.report_serve_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in dir(config):
        if arg.startswith("__"):
            continue
        arg_name = f"--{arg.lower().replace('_', '-')}"
        default_value = getattr(config, arg)
        if type(default_value) in [int, float, str]:
            parser.add_argument(arg_name, type=type(default_value), default=default_value,
                                help=f"Set {arg} (default: {default_value})")
        elif default_value is None:
            parser.add_argument(arg_name, type=int, default=default_value,
                                help=f"Set {arg} (default: {default_value})")
        elif type(default_value) is bool:
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default_value,
                                help=f"Enable or disable {arg} (default: {default_value})")
        elif type(default_value) is list:
            parser.add_argument(arg_name, nargs="*", default=default_value,
                                help=f"Set {arg} as a space-separated list (default: {default_value})")
    args = parser.parse_args()
    run_reporting(args)
