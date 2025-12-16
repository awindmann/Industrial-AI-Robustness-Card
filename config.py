# Reporting-only configuration

# MLflow tracking location (local mlruns-style path or HTTP URI)
LOGDIR = "logs"

# MLflow experiment prefix to search (experiments are expected to be named "{prefix}-{dataset}")
MLFLOW_EXPERIMENT_PREFIX = "card-02"

# Dataset filter defaults (keys from data/datasets/specs.py)
DATA_FILES = ["IndPenSim"]
DATA_TARGETS = []

# Model filter ("all" to include every architecture logged in the experiment)
MODEL = "all"

# Reporting output
REPORT_SERVE = True
REPORT_SERVE_PORT = 8050

# KDE mass to render for the ODD density overlays
ODD_KDE_MASS = 0.98
