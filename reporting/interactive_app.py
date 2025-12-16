from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import numpy as np
from dash import Dash, Input, Output, State, dcc, html
import re
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
import mlflow

from reporting.builder import ReportingBundle, RunReport
from reporting.forecast import ForecastSample, load_forecast_samples
from reporting.metrics import metric_summary_rows, quantile_wis_entry, r_mean_value
from reporting.model_card_content import (
    DATASET_TARGET_LABELS,
    MODEL_CARD_META,
    MODEL_CARD_TEXT,
    NarrativeBlock,
    TARGET_ALIAS_LABELS,
)
from visualizations.plots import (
    plot_feature_kde,
    plot_quantile_forecast,
    plot_scenario_radar,
    plot_split_drift_bar,
)


pio.templates.default = "plotly_white+presentation"



def _dataset_options(bundle: ReportingBundle) -> List[dict]:
    options: List[dict] = []
    for run_id, report in sorted(bundle.runs.items(), key=lambda item: (
        item[1].record.run.data.tags.get("dataset") or "",
        item[1].record.run.data.tags.get("model_architecture") or "",
        item[0],
    )):
        run = report.record.run
        model_arch = run.data.tags.get("model_architecture") or "Model"
        # run_name = run.data.tags.get("mlflow.runName") or run_id[:8]
        full_results = report.tables.get("full_results")
        badge_text = ""
        if isinstance(full_results, pd.DataFrame) and not full_results.empty:
            row = full_results[full_results["run_id"] == run_id]
            if not row.empty and str(row.iloc[0].get("best_model_tag")).lower() == "true":
                badge_text = "(best)"
        label_text = f"{model_arch} {run_id[:8]}"
        if badge_text:
            label_text = f"{label_text} {badge_text}"
        options.append({"label": label_text, "value": run_id})
    return options


def _info_table(items: List[tuple[str, str]]) -> html.Table:
    rows = []
    for label, value in items:
        if value is None:
            continue
        value_str = str(value)
        if not value_str:
            continue
        rows.append(
            html.Tr(
                [
                    html.Th(label, className="info-th"),
                    html.Td(value_str, className="info-td", title=value_str),
                ]
            )
        )
    if not rows:
        return html.Table(
            html.Tbody([html.Tr(html.Td("No data available.", colSpan=2))]),
            className="info-table",
        )
    return html.Table(html.Tbody(rows), className="info-table")


def _render_text_with_links(text: str):
    if not isinstance(text, str) or not text:
        return text
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    parts = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(text[last:m.start()])
        label, url = m.group(1), m.group(2)
        parts.append(html.A(label, href=url, target="_blank", rel="noopener noreferrer"))
        last = m.end()
    if last == 0:
        return text
    if last < len(text):
        parts.append(text[last:])
    return parts


def _slugify_label(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", label.strip().lower())
    return text.strip("-") or "section"


def _tab_entries(block: NarrativeBlock | None) -> List[dict[str, str]]:
    """Flatten narrative sections (or bullets fallback) into nav-ready entries."""
    if block is None:
        return []

    if block.sections:
        entries: List[dict[str, str]] = []
        for section in block.sections:
            label = (section.title or section.key).strip()
            entries.append(
                {
                    "label": label,
                    "body": (section.body or "").strip(),
                    "bullets": list(section.bullets or []),
                    "value": section.key or _slugify_label(label),
                }
            )
        return entries

    entries: List[dict[str, str]] = []
    for bullet in block.bullets:
        if not isinstance(bullet, str):
            continue
        label = bullet.strip()
        if not label:
            continue
        entries.append(
            {
                "label": label,
                "body": "",
                "bullets": [],
                "value": _slugify_label(label),
            }
        )
    return entries


def _tab_config(block: NarrativeBlock | None, preferred_prefix: str | None = None):
    """Return entries, lookup, default value, and initial copy component."""
    entries = _tab_entries(block)
    lookup = {entry["value"]: entry for entry in entries}
    default_value = None
    preferred_prefix = (preferred_prefix or "").lower().strip()
    if entries:
        default_value = next(
            (
                entry["value"]
                for entry in entries
                if preferred_prefix and entry["label"].lower().startswith(preferred_prefix)
            ),
            entries[0]["value"],
        )
    initial_copy = _render_dataset_tab_copy(lookup.get(default_value))
    return entries, lookup, default_value, initial_copy


def _visibility_style(is_visible: bool, display: str = "flex") -> dict:
    return {"display": display} if is_visible else {"display": "none"}


def _render_dataset_tab_copy(entry: dict[str, str] | None):
    if not entry:
        return html.Div("No dataset details available.", className="text-muted dataset-tab-body")
    body_text = entry.get("body") or ""
    bullets = entry.get("bullets") or []
    children = []
    if body_text:
        children.append(html.Div(_render_text_with_links(body_text), className="dataset-tab-body"))
    if bullets:
        children.append(
            html.Ul(
                [html.Li(_render_text_with_links(b)) for b in bullets],
                className="dataset-tab-list",
            )
        )
    if not children:
        children.append(html.Div("Details forthcoming.", className="dataset-tab-body"))
    return html.Div(children)


def _narrative_card(block: NarrativeBlock | None, extra_class: str = "") -> html.Div:
    if block is None:
        return html.Div("No narrative provided.", className=f"narrative-card {extra_class}".strip())

    bullet_items = []
    for entry in block.bullets:
        bullet_items.append(html.Li(_render_text_with_links(entry)))

    children = []
    if isinstance(block.body, str) and block.body.strip():
        children.append(html.P(_render_text_with_links(block.body), className="narrative-body"))
    if bullet_items:
        children.append(html.Ul(bullet_items, className="narrative-list"))
    if block.footer_note:
        children.append(html.Small(block.footer_note, className="narrative-footer"))

    if not children:
        return html.Div(None, className=f"narrative-card {extra_class}".strip())
    return html.Div(children, className=f"narrative-card {extra_class}".strip())


def _format_sample_option(sample: ForecastSample) -> str:
    severity = sample.severity
    severity_text = (
        f"{float(severity):.3f}"
        if severity is not None and np.isfinite(severity)
        else "n/a"
    )
    return f"Sample {sample.sample_id} · {sample.scenario_label} (severity {severity_text})"


def _metric_table(baseline_row: pd.Series | None) -> html.Table:
    if baseline_row is None or baseline_row.empty:
        return html.Table(
            html.Tbody([html.Tr(html.Td("Baseline metrics unavailable.", colSpan=2))]),
            className="metrics-table",
        )
    metric_rows = metric_summary_rows(baseline_row)
    if not metric_rows:
        return html.Table(
            html.Tbody([html.Tr(html.Td("Baseline metrics unavailable.", colSpan=2))]),
            className="metrics-table",
        )
    header = html.Thead(
        html.Tr(
            [
                html.Th("Metric", className="metrics-th metrics-th-label"),
                html.Th("Validation", className="metrics-th metrics-th-value"),
                html.Th("Test", className="metrics-th metrics-th-value"),
                html.Th("Scenarios", className="metrics-th metrics-th-value"),
            ]
        )
    )
    def _format_value(value):
        if value is None or pd.isna(value):
            return "–"
        return f"{float(value):.4f}"

    body_rows = []
    for metric_name, (val, test, pert) in metric_rows:
        body_rows.append(
            html.Tr(
                [
                    html.Th(metric_name, className="metrics-th"),
                    html.Td(_format_value(val), className="metrics-td"),
                    html.Td(_format_value(test), className="metrics-td"),
                    html.Td(_format_value(pert), className="metrics-td"),
                ]
            )
        )
    wis_entry = quantile_wis_entry(baseline_row)
    if wis_entry:
        wis_label, wis_clean, wis_pert = wis_entry
        body_rows.append(
            html.Tr(
                [
                    html.Th(wis_label, className="metrics-th"),
                    html.Td("–", className="metrics-td"),
                    html.Td(_format_value(wis_clean), className="metrics-td"),
                    html.Td(_format_value(wis_pert), className="metrics-td"),
                ]
            )
        )
    r_mean = r_mean_value(baseline_row)
    if r_mean is not None:
        body_rows.append(
            html.Tr(
                [
                    html.Th(["Robustness", html.Br(), "Score (Mean)"], className="metrics-th"),
                    html.Td(_format_value(r_mean), className="metrics-td", colSpan=3),
                ]
            )
        )
    # r_mult_value = baseline_row.get("R_mult")
    # if r_mult_value is not None and not pd.isna(r_mult_value):
    #     body_rows.append(
    #         html.Tr(
    #             [
    #                 html.Th(["Robustness", html.Br(), "Score (Product)"], className="metrics-th"),
    #                 html.Td(f"{float(r_mult_value):.4f}", className="metrics-td", colSpan=3),
    #             ]
    #         )
    #     )
    return html.Table([header, html.Tbody(body_rows)], className="metrics-table")


def _format_timestamp(epoch_ms: int | None) -> str | None:
    if epoch_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    except (OSError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _format_window(params: dict) -> str | None:
    input_len = params.get("input_len")
    target_len = params.get("target_len")
    if input_len is None or target_len is None:
        return None
    return f"{input_len} → {target_len}"


def _format_split_text(params: dict) -> str | None:
    train_split = params.get("train_split")
    val_split = params.get("val_split")
    if train_split is None and val_split is None:
        return None
    try:
        train_pct = float(train_split) * 100 if train_split is not None else None
        val_pct = float(val_split) * 100 if val_split is not None else None
        test_pct = None
        if train_pct is not None and val_pct is not None:
            test_pct = max(0.0, 100.0 - train_pct - val_pct)
    except (TypeError, ValueError):
        return None

    pieces = []
    if train_pct is not None:
        pieces.append(f"Train {train_pct:.0f}%")
    if val_pct is not None:
        pieces.append(f"Val {val_pct:.0f}%")
    if test_pct is not None:
        pieces.append(f"Test {test_pct:.0f}%")
    if not pieces:
        return None
    return ", ".join(pieces)


def _normalize_lookup_value(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _humanize_channel_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return ""
    return text.replace("_", " ")


def _format_channel_tokens(value: str | None) -> str | None:
    if not value:
        return None
    raw_tokens = str(value).replace(",", ";").split(";")
    tokens = []
    for token in raw_tokens:
        cleaned = _humanize_channel_name(token)
        if cleaned:
            tokens.append(cleaned)
    if not tokens:
        return None
    if len(tokens) <= 3:
        return ", ".join(tokens)
    extra = len(tokens) - 3
    return f"{', '.join(tokens[:3])} +{extra} more"


def _resolve_target_label(run, baseline_row: pd.Series | None) -> str | None:
    alias_candidates = []
    if baseline_row is not None:
        alias_candidates.append(baseline_row.get("target_alias"))
    alias_candidates.extend(
        [
            run.data.tags.get("target_alias"),
            run.data.params.get("target_alias"),
            run.data.params.get("target"),
        ]
    )
    for alias_candidate in alias_candidates:
        alias_norm = _normalize_lookup_value(alias_candidate)
        if not alias_norm:
            continue
        label = TARGET_ALIAS_LABELS.get(alias_norm)
        if label:
            return label
        return str(alias_candidate).strip()

    dataset_candidates = []
    if baseline_row is not None:
        dataset_candidates.append(baseline_row.get("dataset"))
    dataset_candidates.extend(
        [
            run.data.tags.get("dataset"),
            run.data.params.get("dataset"),
        ]
    )
    for dataset_candidate in dataset_candidates:
        dataset_norm = _normalize_lookup_value(dataset_candidate)
        if not dataset_norm:
            continue
        label = DATASET_TARGET_LABELS.get(dataset_norm)
        if label:
            return label

    channel_candidates = []
    if baseline_row is not None:
        channel_candidates.append(baseline_row.get("target_channels"))
    channel_candidates.extend(
        [
            run.data.tags.get("target_channels"),
            run.data.params.get("target_channels"),
        ]
    )
    for channel_candidate in channel_candidates:
        formatted = _format_channel_tokens(channel_candidate)
        if formatted:
            return formatted
    return None


def _resolve_loss_function(run, baseline_row: pd.Series | None) -> str | None:
    keys = ["loss_fn", "loss_function", "criterion", "loss", "objective", "test_metric"]
    if baseline_row is not None:
        for key in keys:
            value = baseline_row.get(key)
            if value:
                return str(value)
    for key in keys:
        value = run.data.params.get(key)
        if value:
            return str(value)
    for key in keys:
        value = run.data.tags.get(key)
        if value:
            return str(value)
    return None


def _resolve_run_timestamp(run, baseline_row: pd.Series | None) -> str | None:
    test_time = None
    if baseline_row is not None:
        test_time = baseline_row.get("test_end_time")
    if test_time is None:
        test_time = run.data.tags.get("test_end_time") or run.data.params.get("test_end_time")
    train_time = run.info.end_time or run.info.start_time
    return _format_timestamp(test_time or train_time)


def _general_info_list(run, baseline_row: pd.Series | None, report: RunReport) -> html.Table:
    loss_fn = _resolve_loss_function(run, baseline_row)
    windowing = _format_window(run.data.params)
    timestamp = _resolve_run_timestamp(run, baseline_row)
    train_commit = (
        baseline_row.get("train_commit") if baseline_row is not None else run.data.tags.get("train_commit")
    )
    # splits = _format_split_text(run.data.params)
    provider = run.data.tags.get("provider") or MODEL_CARD_META.get("provider")
    contact = run.data.tags.get("provider_contact") or MODEL_CARD_META.get("contact")
    model_label = (
        (baseline_row.get("model") if baseline_row is not None else None)
        or run.data.tags.get("model_architecture")
        or run.data.params.get("model_architecture")
        or run.data.params.get("model")
        or run.data.tags.get("model_loader")
    )
    target_label = _resolve_target_label(run, baseline_row)
    # Order by manufacturer relevance: primary context → training config → traceability
    dataset_value = (
        (baseline_row.get("dataset") if baseline_row is not None else None)
        or run.data.tags.get("dataset")
        or run.data.params.get("dataset")
    )
    items = [
        # ("Commit", train_commit),
        ("Model", model_label),
        ("Dataset", dataset_value),
        ("Target", target_label),
        ("Loss Function", loss_fn),
        ("Window", windowing),
        ("Date", timestamp),
        # ("Data Split", splits),
        # ("Experiment", report.record.experiment.name),
        ("Model ID", run.info.run_id),
        ("Provider", provider),
        ("Contact", contact),
        # ("Seed", run.data.params.get("seed")),
    ]
    return _info_table(items)


def _blank_figure(message: str | None = None, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=25, t=40, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if message:
        fig.add_annotation(
            text=message,
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
        )
    return fig


def create_dash_app(bundle: ReportingBundle) -> Dash:
    mlflow.set_tracking_uri(bundle.tracking_uri)
    mlflow_client = mlflow.MlflowClient()
    forecast_cache: Dict[str, dict] = {}
    odd_kde_mass = float(getattr(bundle, "odd_kde_mass", 0.98))
    if not (0.0 < odd_kde_mass < 1.0):
        raise ValueError(f"Invalid odd_kde_mass {odd_kde_mass}; must be in (0,1).")

    external_stylesheets = [dbc.themes.LUX]
    assets_path = Path(__file__).resolve().parents[1] / "assets"
    app = Dash(__name__, external_stylesheets=external_stylesheets, assets_folder=str(assets_path))

    run_selector = dcc.Dropdown(
        id="run-selector",
        options=_dataset_options(bundle),
        value=next(iter(bundle.runs.keys())) if bundle.runs else None,
        clearable=False,
        style={"width": "100%"},
    )

    header_section = html.Div(
        [
            html.Div(
                [
                    html.H1(id="hero-title", className="hero-title"),
                    html.P(id="hero-subtitle", className="hero-subtitle"),
                ],
                className="header-copy compact",
            ),
            html.Div(
                [
                    html.Label("Run", className="run-label"),
                    html.Div(run_selector, className="run-selector"),
                    html.Button("Export PDF", id="export-button", className="export-btn"),
                ],
                className="header-controls",
            ),
        ],
        className="report-header panel",
    )

    general_intended_row = html.Div(
        [
            html.Div(
                [
                    html.Div("General Information", className="section-title with-icon"),
                    _narrative_card(MODEL_CARD_TEXT.get("general_information")),
                    html.Div(id="general-info-body", className="info-body"),
                ],
                className="panel info-panel",
            ),
            html.Div(
                [
                    html.Div("Intended Use", className="section-title with-icon"),
                    _narrative_card(MODEL_CARD_TEXT.get("intended_use")),
                ],
                className="panel intended-panel narrative-panel",
            ),
        ],
        className="general-intended-row",
    )

    dataset_tab_entries, dataset_tab_lookup, dataset_tab_default_value, dataset_tab_copy_initial = _tab_config(
        MODEL_CARD_TEXT.get("dataset_spec"),
        preferred_prefix="distribution",
    )
    dataset_visual_value = dataset_tab_default_value
    scenario_visual_value = "scenario-catalog"

    evaluation_tab_entries, evaluation_tab_lookup, evaluation_tab_default_value, evaluation_tab_copy_initial = _tab_config(
        MODEL_CARD_TEXT.get("evaluation"),
        preferred_prefix="robustness",
    )
    robustness_visual_value = next(
        (
            entry["value"]
            for entry in evaluation_tab_entries
            if entry["label"].lower().startswith("robustness")
        ),
        None,
    )
    uq_visual_value = next(
        (
            entry["value"]
            for entry in evaluation_tab_entries
            if entry["label"].lower().startswith("uncertainty")
        ),
        None,
    )

    dataset_panel = html.Div(
        [
            html.Div("Data", className="section-title with-icon"),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="dataset-key-dropdown",
                                options=[
                                    {"label": entry["label"], "value": entry["value"]}
                                    for entry in dataset_tab_entries
                                ],
                                value=dataset_tab_default_value,
                                clearable=False,
                                className="dataset-nav-dropdown",
                            ),
                            html.Div(
                                dataset_tab_copy_initial,
                                id="dataset-tab-copy",
                                className="dataset-tab-copy",
                            ),
                        ],
                        className="dataset-nav-pane",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Tabs(
                                        id="dataset-tabs",
                                        value="kde",
                                        className="dataset-tabs",
                                        children=[
                                            dcc.Tab(
                                                label="Feature KDE",
                                                value="kde",
                                                className="dataset-tab",
                                                selected_className="dataset-tab--selected",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dcc.Dropdown(
                                                                        id="kde-x-selector",
                                                                        options=[],
                                                                        value=None,
                                                                        clearable=False,
                                                                        placeholder="X feature",
                                                                        style={"flex": 1, "minWidth": "120px"},
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="kde-y-selector",
                                                                        options=[],
                                                                        value=None,
                                                                        clearable=False,
                                                                        placeholder="Y feature",
                                                                        style={"flex": 1, "minWidth": "120px"},
                                                                    ),
                                                                ],
                                                                className="selector-inline drift-selector",
                                                                style={"gap": "8px", "marginTop": "8px"},
                                                            ),
                                                            dcc.Graph(
                                                                id="feature-kde-graph",
                                                                className="graph-block dataset-graph",
                                                                figure=_blank_figure("KDE loading", height=200),
                                                                config={"displaylogo": False, "displayModeBar": False},
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                            ),
                                            dcc.Tab(
                                                label="Drift Ranking",
                                                value="drift",
                                                className="dataset-tab",
                                                selected_className="dataset-tab--selected",
                                                children=[
                                                    html.Div(
                                                        [
                                                            dcc.Dropdown(
                                                                id="drift-focus",
                                                                options=[],
                                                                value=None,
                                                                clearable=False,
                                                                style={"flex": 1},
                                                            ),
                                                        ],
                                                        className="selector-inline drift-selector",
                                                    ),
                                                    dcc.Graph(
                                                        id="dataset-drift-graph",
                                                        className="graph-block dataset-graph",
                                                        figure=_blank_figure("Distribution profile loading", height=200),
                                                        config={"displaylogo": False, "displayModeBar": False},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    )
                                ],
                                id="distribution-visuals",
                                className="dataset-drift-card distribution-visuals",
                                style=_visibility_style(
                                    dataset_tab_default_value is not None
                                    and dataset_tab_default_value == dataset_visual_value
                                ),
                            ),
                            html.Div(
                                [
                                    html.Img(
                                        src="/assets/scenarios_wide.png",
                                        alt="Scenario catalog overview",
                                        style={"width": "100%", "height": "auto", "maxHeight": "280px", "objectFit": "contain"},
                                    ),
                                ],
                                id="scenario-visuals",
                                className="dataset-drift-card scenario-visuals",
                                style=_visibility_style(False),
                            ),
                        ],
                        className="dataset-visuals-pane",
                    ),
                ],
                className="dataset-content",
            ),
        ],
        className="panel dataset-panel narrative-panel",
    )

    evaluation_panel = html.Div(
        [
            html.Div("Evaluation", className="section-title with-icon"),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="evaluation-key-dropdown",
                                options=[
                                    {"label": entry["label"], "value": entry["value"]}
                                    for entry in evaluation_tab_entries
                                ],
                                value=evaluation_tab_default_value,
                                clearable=False,
                                className="evaluation-nav-dropdown",
                            ),
                            html.Div(id="kpi-metrics", className="metrics-table-container"),
                            html.Div(
                                evaluation_tab_copy_initial,
                                id="evaluation-tab-copy",
                                className="dataset-tab-copy",
                            ),
                        ],
                        className="evaluation-nav-pane",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Scenario Robustness", className="section-subtitle"),
                                            dcc.Graph(
                                                id="radar-graph",
                                                className="graph-block dataset-graph eval-radar-graph",
                                                figure=_blank_figure("Scenario metrics loading", height=200),
                                                config={"displaylogo": False, "displayModeBar": False},
                                            ),
                                        ],
                                        id="robustness-visuals",
                                        className="dataset-drift-card evaluation-visuals",
                                        style=_visibility_style(
                                            robustness_visual_value is not None
                                            and evaluation_tab_default_value == robustness_visual_value
                                        ),
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Quantile Forecast", className="section-subtitle"),
                                            html.Div(
                                                [
                                                    html.Label("Sample", className="form-label fw-semibold mb-0"),
                                                    dcc.Dropdown(
                                                        id="sample-selector",
                                                        options=[],
                                                        value=None,
                                                        clearable=False,
                                                        style={"flex": 1},
                                                    ),
                                                ],
                                                className="selector-inline",
                                            ),
                                            html.Div(id="forecast-context", className="text-muted small mb-2"),
                                            dcc.Graph(
                                                id="forecast-graph",
                                                className="graph-block dataset-graph",
                                                figure=_blank_figure("Forecast loading", height=200),
                                                config={"displaylogo": False, "displayModeBar": False},
                                            ),
                                        ],
                                        id="uq-visuals",
                                        className="dataset-drift-card evaluation-visuals",
                                        style=_visibility_style(
                                            uq_visual_value is not None
                                            and evaluation_tab_default_value == uq_visual_value
                                        ),
                                    ),
                                ],
                            ),
                        ],
                        className="evaluation-visuals-pane",
                    ),
                ],
                className="evaluation-content",
            ),
        ],
        className="panel evaluation-panel narrative-panel",
    )

    limitations_panel = html.Div(
        [
            html.Div("Limitations", className="section-title with-icon"),
            _narrative_card(MODEL_CARD_TEXT.get("limitations")),
        ],
        className="panel limitations-panel narrative-panel",
    )

    app.layout = html.Div(
        [
            html.Div(
                [
                    header_section,
                    html.Div(
                        [
                            general_intended_row,
                            dataset_panel,
                            evaluation_panel,
                            limitations_panel,
                        ],
                        className="model-card shadow-lg",
                    ),
                ],
                className="report-container",
            ),
            html.Div(id="print-trigger", style={"display": "none"}),
            dcc.Store(id="dataset-profile-store"),
            dcc.Store(id="dataset-kde-store"),
            dcc.Store(id="kde-visible-store"),
            dcc.Store(id="pdf-title-store"),
        ],
        className="report-shell",
    )

    @app.callback(
        Output("hero-title", "children"),
        Output("hero-subtitle", "children"),
        Output("general-info-body", "children"),
        Output("kpi-metrics", "children"),
        Output("radar-graph", "figure"),
        Output("sample-selector", "options"),
        Output("sample-selector", "value"),
        Output("sample-selector", "disabled"),
        Output("dataset-profile-store", "data"),
        Output("drift-focus", "options"),
        Output("drift-focus", "value"),
        Output("drift-focus", "disabled"),
        Output("dataset-kde-store", "data"),
        Output("kde-x-selector", "options"),
        Output("kde-x-selector", "value"),
        Output("kde-x-selector", "disabled"),
        Output("kde-y-selector", "options"),
        Output("kde-y-selector", "value"),
        Output("kde-y-selector", "disabled"),
        Output("kde-visible-store", "data"),
        Output("pdf-title-store", "data"),
        Input("run-selector", "value"),
    )
    def _update_run_view(run_id):
        report: RunReport = bundle.runs.get(run_id)
        if report is None:
            empty_fig = _blank_figure("Select a run", height=320)
            return (
                "No run selected",
                "",
                html.Div("Select a run to populate the report.", className="text-muted"),
                html.Div("No metrics available.", className="text-muted"),
                empty_fig,
                [],
                None,
                True,
                None,
                [],
                None,
                True,
                None,
                [],
                None,
                True,
                [],
                None,
                True,
                "Model Card",
            )

        run = report.record.run
        full_results = report.tables.get("full_results")
        dataset = run.data.tags.get("dataset")
        model = run.data.tags.get("model_architecture")
        run_name = run.data.tags.get("mlflow.runName") or run.info.run_id

        model_label = model or run.data.params.get("model_architecture") or run.data.params.get("model")
        if not model_label and full_results is not None and not full_results.empty and "model" in full_results.columns:
            model_label = full_results["model"].iloc[0]
        dataset_label = dataset or run.data.params.get("dataset")
        if not dataset_label and full_results is not None and not full_results.empty and "dataset" in full_results.columns:
            dataset_label = full_results["dataset"].iloc[0]
        hero_title_text = model_label or run_name
        hero_subtitle_text = dataset_label or report.record.experiment.name

        general_info_component = _general_info_list(run, None, report)
        baseline_metrics_component = html.Div("No baseline metrics available.", className="text-muted")
        radar_fig = _blank_figure("No scenario metrics", height=210)
        sample_options: List[dict] = []
        sample_value = None
        forecast_cache.pop(run_id, None)
        profile_store_data = None
        drift_focus_options: List[dict] = []
        drift_focus_value = None
        drift_focus_disabled = True
        kde_store_data = None
        data_kde = report.tables.get("data_kde")
        kde_feature_options: List[dict] = []
        kde_x_value = None
        kde_y_value = None
        kde_x_disabled = True
        kde_y_disabled = True

        baseline_df = full_results[full_results["stage"] == "train"] if full_results is not None else None

        baseline_row = None
        radar_traces = {}
        default_visible_kde: List[str] = []
        if baseline_df is not None and not baseline_df.empty:
            scenario_columns = [col for col in baseline_df.columns if col.startswith("scenario_")]
            for _, row in baseline_df.iterrows():
                if row.get("run_id") == run.info.run_id:
                    baseline_row = row
                base_label = row.get("model") or row.get("model_architecture") or row.get("backbone_architecture") or "Model"
                label = f"{base_label} {str(row.get('run_id'))[:8]}"
                scenario_scores = {
                    col[len("scenario_"):].replace("_", " ").title(): float(row[col])
                    for col in scenario_columns
                    if pd.notna(row[col])
                }
                if scenario_scores:
                    radar_traces[label] = scenario_scores
            if radar_traces:
                radar_fig = plot_scenario_radar(radar_traces, title=None)
            if baseline_row is not None:
                general_info_component = _general_info_list(run, baseline_row, report)
                baseline_metrics_component = _metric_table(baseline_row)

        if data_kde:
            default_visible_kde = [key for key in ("train", "test") if key in data_kde]
        radar_fig.update_layout(
            height=210,
            margin=dict(l=22, r=22, t=24, b=28),
            title=None,
            font=dict(size=16),
            showlegend=False,
            polar=dict(
                radialaxis=dict(tickfont=dict(size=13)),
                angularaxis=dict(tickfont=dict(size=14)),
            ),
        )

        quantile_rows = (
            full_results[
                (full_results["stage"] == "uq")
                & (full_results["linked_backbone_run"] == run.info.run_id)
            ]
            if full_results is not None
            else pd.DataFrame()
        )
        if baseline_row is not None and not quantile_rows.empty:
            samples = load_forecast_samples(mlflow_client, baseline_row, quantile_rows)
            if samples:
                sample_options = [
                    {
                        "label": _format_sample_option(sample),
                        "value": sample.sample_id,
                    }
                    for sample in samples
                ]
                sample_value = sample_options[0]["value"]
                forecast_cache[run_id] = {
                    "model_label": model or baseline_row.get("model") or "Model",
                    "samples": samples,
                }

        sample_disabled = not sample_options

        profile_df = report.tables.get("data_profile")
        if profile_df is not None and not profile_df.empty:
            profile_store_data = profile_df.to_dict("records")
            drift_focus_options = [{"label": "Train vs Test", "value": "train_test"}]
            if profile_df["score_val_test"].notna().any():
                drift_focus_options.append({"label": "Val vs Test", "value": "val_test"})
                drift_focus_options.append({"label": "Train + Val vs Test", "value": "both"})
            drift_focus_value = drift_focus_options[0]["value"]
            drift_focus_disabled = False

        if data_kde:
            kde_store_data = data_kde
            feature_sets = [
                set(features.keys())
                for features in data_kde.values()
                if isinstance(features, dict)
            ]
            base_features: List[str] = sorted(set.intersection(*feature_sets)) if feature_sets else []

            feature_names = list(dict.fromkeys(base_features))
            if profile_df is not None and not profile_df.empty and "score_train_test" in profile_df.columns:
                df_scores = profile_df[["feature", "score_train_test"]].copy()
                df_scores = df_scores[df_scores["feature"].isin(feature_names)]
                df_scores = df_scores.sort_values("score_train_test", ascending=False)
                feature_names = list(df_scores["feature"])

            feature_options = [{"label": name, "value": name} for name in feature_names]
            if feature_options:
                kde_feature_options = feature_options
                kde_x_value = feature_options[0]["value"]
                kde_y_value = feature_options[1]["value"] if len(feature_options) > 1 else feature_options[0]["value"]
                kde_x_disabled = False
                kde_y_disabled = False

        return (
            hero_title_text,
            hero_subtitle_text,
            general_info_component,
            baseline_metrics_component,
            radar_fig,
            sample_options,
            sample_value,
            sample_disabled,
            profile_store_data,
            drift_focus_options,
            drift_focus_value,
            drift_focus_disabled,
            kde_store_data,
            kde_feature_options,
            kde_x_value,
            kde_x_disabled,
            kde_feature_options,
            kde_y_value,
            kde_y_disabled,
            default_visible_kde,
            f"Model Card - {run_name}",
        )

    @app.callback(
        Output("dataset-tab-copy", "children"),
        Output("distribution-visuals", "style"),
        Output("scenario-visuals", "style"),
        Input("dataset-key-dropdown", "value"),
    )
    def _update_dataset_tab(selected_tab):
        entry = dataset_tab_lookup.get(selected_tab)
        show_distribution = dataset_visual_value is not None and selected_tab == dataset_visual_value
        show_scenario = selected_tab == scenario_visual_value
        return (
            _render_dataset_tab_copy(entry),
            _visibility_style(show_distribution),
            _visibility_style(show_scenario),
        )

    @app.callback(
        Output("evaluation-tab-copy", "children"),
        Output("robustness-visuals", "style"),
        Output("uq-visuals", "style"),
        Output("kpi-metrics", "style"),
        Input("evaluation-key-dropdown", "value"),
    )
    def _update_evaluation_tab(selected_tab):
        entry = evaluation_tab_lookup.get(selected_tab)
        show_robustness = robustness_visual_value is not None and selected_tab == robustness_visual_value
        show_uq = uq_visual_value is not None and selected_tab == uq_visual_value
        return (
            _render_dataset_tab_copy(entry),
            _visibility_style(show_robustness),
            _visibility_style(show_uq),
            _visibility_style(show_robustness, display="block"),
        )

    @app.callback(
        Output("forecast-graph", "figure"),
        Output("forecast-context", "children"),
        Input("run-selector", "value"),
        Input("sample-selector", "value"),
    )
    def _update_forecast(run_id, sample_id):
        if run_id is None:
            return _blank_figure("Select a run to view forecasts.", height=320), ""

        cache_entry = forecast_cache.get(run_id)
        if not cache_entry:
            return _blank_figure("No quantile forecasts available for this run.", height=320), ""

        samples = cache_entry["samples"]
        sample_lookup: Dict[int, ForecastSample] = {entry.sample_id: entry for entry in samples}
        selected_sample = sample_lookup.get(sample_id) if sample_id is not None else samples[0]
        if selected_sample is None:
            raise ValueError(f"Sample {sample_id} not found in forecast cache.")

        figure = plot_quantile_forecast(
            selected_sample.output_time,
            selected_sample.target,
            selected_sample.prediction,
            quantile_predictions=selected_sample.quantile_predictions or None,
            clean_input=selected_sample.clean_input,
            perturbed_input=selected_sample.perturbed_input,
            input_time_index=selected_sample.input_time,
            input_feature_names=selected_sample.input_feature_names,
            target_feature_names=selected_sample.target_feature_names,
            affected_feature_names=selected_sample.affected_feature_names,
            title=f"Quantile Forecast for {cache_entry['model_label']}",
            scenario=selected_sample.scenario_label,
            severity=selected_sample.severity,
        )
        figure.update_layout(
            height=320,
            margin=dict(l=30, r=25, t=30, b=28),
            showlegend=False,
            hovermode="x unified",
        )

        return figure, ""

    @app.callback(
        Output("dataset-drift-graph", "figure"),
        Input("dataset-profile-store", "data"),
        Input("drift-focus", "value"),
    )
    def _update_dataset_drift(profile_data, focus):
        if not profile_data:
            return _blank_figure("No distribution profile available.", height=320)
        df = pd.DataFrame(profile_data)
        fig = plot_split_drift_bar(df, focus=focus or "train_test", height=200)
        fig.update_layout(margin=dict(l=16, r=16, t=12, b=16))
        return fig

    @app.callback(
        Output("feature-kde-graph", "figure"),
        Input("dataset-kde-store", "data"),
        Input("kde-x-selector", "value"),
        Input("kde-y-selector", "value"),
        Input("kde-visible-store", "data"),
    )
    def _update_feature_kde(kde_data, feature_x, feature_y, visible_defaults):
        if not kde_data or not feature_x or not feature_y:
            return _blank_figure("No distribution profile available.", height=200)
        fig = plot_feature_kde(
            feature_x,
            feature_y,
            kde_data,
            visible_scenarios=visible_defaults or None,
            odd_kde_mass=odd_kde_mass,
            height=200,
        )
        fig.update_layout(margin=dict(l=16, r=16, t=12, b=16))
        return fig

    app.clientside_callback(
        """
        function(n_clicks, title){
            if(!n_clicks){return ""}
            var original = document.title;
            if(title){ document.title = title; }
            setTimeout(function(){ window.print(); }, 0);
            setTimeout(function(){ document.title = original; }, 1000);
            return "";
        }
        """,
        Output("print-trigger", "children"),
        Input("export-button", "n_clicks"),
        State("pdf-title-store", "data"),
        prevent_initial_call=True,
    )

    return app


def launch_dash_app(bundle: ReportingBundle, port: int = 8050):
    app = create_dash_app(bundle)
    app.run(port=port, debug=False)
