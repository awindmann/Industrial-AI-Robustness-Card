from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

TargetLike = Union[str, Sequence[str], None]


def _normalize_alias(alias: str) -> str:
    return alias.strip().lower()


def _normalize_path(path: str) -> str:
    path_str = str(path)
    if path_str.lower().startswith("s3:"):
        _, _, suffix = path_str.partition(":")
        return "s3://" + suffix.lstrip("/")
    return Path(path_str).as_posix()


@dataclass(frozen=True)
class ResolvedDatasetSpec:
    key: str
    path: str
    input_channels: Optional[Tuple[str, ...]]
    target_channels: Optional[Tuple[str, ...]]
    target_alias: Optional[str]
    description: Optional[str] = None
    batch_column: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        if self.input_channels is not None and not self.input_channels:
            raise ValueError("input_channels cannot be empty; use None for 'all'.")
        if self.target_channels is not None and not self.target_channels:
            raise ValueError("target_channels cannot be empty; use None for 'all'.")
        if self.batch_column is not None and not self.batch_column:
            raise ValueError("batch_column must be a non-empty string or None.")

    @property
    def path_posix(self) -> str:
        return _normalize_path(self.path)


@dataclass
class DatasetSpec:
    key: str
    path: str
    input_channels: Optional[Sequence[str]] = None
    channel_groups: Mapping[str, Sequence[str]] = field(default_factory=dict)
    default_target: TargetLike = None
    description: Optional[str] = None
    batch_column: Optional[str] = None
    normalized_path: str = field(init=False)

    def __post_init__(self) -> None:
        normalized_groups = {}
        alias_display = {}
        for alias, columns in (self.channel_groups or {}).items():
            if not alias:
                raise ValueError("Target group aliases must be non-empty strings.")
            norm_alias = _normalize_alias(alias)
            normalized_groups[norm_alias] = tuple(columns)
            alias_display[norm_alias] = alias
        self._target_groups_lookup = normalized_groups
        self._alias_display = alias_display
        normalized_path = _normalize_path(self.path)
        self.path = normalized_path
        self.normalized_path = normalized_path
        self._input_channels_tuple = tuple(self.input_channels) if self.input_channels is not None else None
        if self.batch_column is not None and not self.batch_column:
            raise ValueError("batch_column must be a non-empty string or None.")

    def resolve(self, target: TargetLike = None) -> ResolvedDatasetSpec:
        alias = None
        resolved_target: Optional[Tuple[str, ...]]
        target_value = target if target is not None else self.default_target

        if isinstance(target_value, str):
            norm = _normalize_alias(target_value)
            alias = target_value
            if norm == "all" or not target_value:
                resolved_target = None
            else:
                try:
                    resolved_target = self._target_groups_lookup[norm]
                except KeyError as exc:
                    raise KeyError(
                        f"Unknown target alias '{target_value}' for dataset '{self.key}'. "
                        f"Available: {sorted(self._alias_display.values()) or ['all']}"
                    ) from exc
        elif target_value is None:
            alias = None if self.default_target not in (None, "all") else "all"
            resolved_target = None if self.default_target in (None, "all") else self._coerce_sequence(self.default_target)
        else:
            resolved_target = self._coerce_sequence(target_value)

        return ResolvedDatasetSpec(
            key=self.key,
            path=self.path,
            input_channels=self._input_channels_tuple,
            target_channels=resolved_target,
            target_alias=alias,
            description=self.description,
            batch_column=self.batch_column,
        )

    @staticmethod
    def _coerce_sequence(values: Union[Sequence[str], None]) -> Tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, str):
            return (values,)
        coerced = tuple(values)
        if not coerced:
            raise ValueError("Target channel list must not be empty; use None for 'all'.")
        return coerced

    @property
    def available_target_aliases(self) -> Tuple[str, ...]:
        aliases = tuple(self._alias_display.values())
        return aliases if aliases else ("all",)


class DatasetRegistry:
    def __init__(self, *specs: DatasetSpec):
        self._specs_by_key: Dict[str, DatasetSpec] = {}
        self._specs_by_path: Dict[str, DatasetSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: DatasetSpec) -> None:
        key_norm = _normalize_alias(spec.key)
        if key_norm in self._specs_by_key:
            raise ValueError(f"Duplicate dataset key '{spec.key}'.")
        if spec.normalized_path in self._specs_by_path:
            raise ValueError(f"Duplicate dataset path '{spec.path}'.")
        self._specs_by_key[key_norm] = spec
        self._specs_by_path[spec.normalized_path] = spec

    def keys(self) -> Sequence[str]:
        return tuple(spec.key for spec in self._specs_by_key.values())

    def get(self, identifier: str) -> DatasetSpec:
        norm = _normalize_alias(identifier)
        if norm in self._specs_by_key:
            return self._specs_by_key[norm]
        normalized_path = _normalize_path(identifier)
        if normalized_path in self._specs_by_path:
            return self._specs_by_path[normalized_path]
        raise KeyError(f"Dataset '{identifier}' is not registered.")

    def resolve_many(
        self,
        identifiers: Iterable[str],
        targets: Optional[Sequence[TargetLike]] = None,
    ) -> Tuple[ResolvedDatasetSpec, ...]:
        resolved_specs = []
        overrides = list(targets) if targets is not None else None
        for idx, identifier in enumerate(identifiers):
            override = None
            if overrides is not None and idx < len(overrides):
                override = overrides[idx]
                if isinstance(override, str) and not override:
                    override = None
            resolved_specs.append(self._resolve_with_fallback(identifier, override))
        return tuple(resolved_specs)

    def maybe_get(self, identifier: str) -> Optional[DatasetSpec]:
        try:
            return self.get(identifier)
        except KeyError:
            return None

    def add_alias(self, alias: str, target_key: str) -> None:
        spec = self.get(target_key)
        alias_norm = _normalize_alias(alias)
        if alias_norm in self._specs_by_key:
            raise ValueError(f"Alias '{alias}' already registered.")
        self._specs_by_key[alias_norm] = spec

    def _resolve_with_fallback(self, identifier: str, target_override: TargetLike) -> ResolvedDatasetSpec:
        try:
            spec = self.get(identifier)
        except KeyError:
            return self._build_default_spec(identifier, target_override)
        return spec.resolve(target_override)

    @staticmethod
    def _build_default_spec(identifier: str, target_override: TargetLike) -> ResolvedDatasetSpec:
        if target_override not in (None, "", "all"):
            raise KeyError(
                f"Cannot apply target override '{target_override}' to unknown dataset '{identifier}'."
            )
        identifier_str = _normalize_path(str(identifier))
        if identifier_str.startswith("s3://"):
            inferred_name = Path(identifier_str[5:]).name or identifier_str
            key = Path(inferred_name).stem or identifier_str
            return ResolvedDatasetSpec(
                key=key,
                path=identifier_str,
                input_channels=None,
                target_channels=None,
                target_alias=None,
                description=None,
                batch_column=None,
            )
        path = Path(identifier_str)
        if (
            not path.suffix
            and path.name == identifier_str
            and "/" not in identifier_str
            and "\\" not in identifier_str
        ):
            raise KeyError(
                f"Dataset '{identifier}' is not registered and does not look like a file path."
            )
        key = path.stem or identifier_str
        return ResolvedDatasetSpec(
            key=key,
            path=str(path),
            input_channels=None,
            target_channels=None,
            target_alias=None,
            description=None,
            batch_column=None,
        )


def resolve_with_defaults(
    default_datasets: Sequence[str],
    default_targets: Sequence[TargetLike],
    datasets: Optional[Union[str, Sequence[str]]] = None,
    targets: Optional[Union[TargetLike, Sequence[TargetLike]]] = None,
):
    dataset_list = _ensure_str_list(datasets) if datasets is not None else list(default_datasets)
    if not dataset_list:
        dataset_list = list(default_datasets)
    target_list: Optional[Sequence[TargetLike]]
    if targets is not None:
        target_list = _ensure_target_list(targets)
    elif default_targets:
        target_list = list(default_targets)
    else:
        target_list = None
    from .specs import DATASET_REGISTRY  # local import to avoid cycles

    return DATASET_REGISTRY.resolve_many(dataset_list, target_list)


def spec_to_tags(spec: ResolvedDatasetSpec, *, n_inputs: int, n_outputs: int) -> Mapping[str, str]:
    target_channels = spec.target_channels or ()
    alias = spec.target_alias or ("all" if n_outputs == n_inputs else "custom")
    input_channels = spec.input_channels or ()
    return {
        "dataset_path": spec.path,
        "target_alias": alias,
        "target_channels": ";".join(target_channels),
        "target_channel_count": str(n_outputs),
        "input_channel_count": str(n_inputs),
        "input_channels": ";".join(input_channels),
    }


def filter_spec_tags(tags: Mapping[str, str]):
    keep = {
        "dataset_path",
        "target_alias",
        "target_channels",
        "target_channel_count",
        "input_channels",
        "input_channel_count",
    }
    return {k: tags[k] for k in keep if k in tags}


def _ensure_str_list(values: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]


def _ensure_target_list(values: Union[TargetLike, Sequence[TargetLike]]) -> Sequence[TargetLike]:
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


__all__ = [
    "DatasetSpec",
    "ResolvedDatasetSpec",
    "DatasetRegistry",
    "TargetLike",
    "resolve_with_defaults",
    "spec_to_tags",
    "filter_spec_tags",
]
