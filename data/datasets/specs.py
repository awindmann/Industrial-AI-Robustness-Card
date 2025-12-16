from __future__ import annotations

from pathlib import Path

from .base import DatasetRegistry, DatasetSpec


DATA_ROOT = Path("data/processed")

DATASET_REGISTRY = DatasetRegistry(
    DatasetSpec(
        key="IndPenSim",
        path=str(DATA_ROOT / "indpensim.parquet"),
        input_channels=(
            "Fs_substrate",
            "Fa_acid",
            "Fb_base",
            "F_PAA",
            "F_oil",
            "Fg_aeration",
            "agitator_rpm",
            "Fh_hot_water",
            "Fc_cold_water",
            "Fw_dilution",
            "pH",
            "temperature",
            "dissolved_oxygen",
            "offgas_CO2_pct",
            "offgas_O2_pct",
            "vessel_volume",
            "pressure",
            "control_mode",
            "time_since_inoculation",
        ),
        channel_groups={
            "penicillin": ("penicillin_concentration",),
        },
        default_target="penicillin",
        description=(
            "Industrial penicillin simulation with batch-level time series at 12-minute cadence. "
            "Inputs include actuator commands, sensor signals, and control context."
        ),
        batch_column="batch_id",
    ),
)

for alias, canonical in {
    str(DATA_ROOT / "indpensim.parquet"): "IndPenSim",
    "indpensim": "IndPenSim",
}.items():
    try:
        DATASET_REGISTRY.add_alias(alias, canonical)
    except ValueError:
        pass

__all__ = ["DATASET_REGISTRY", "DatasetSpec"]
