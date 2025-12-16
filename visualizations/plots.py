import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull

from typing import Mapping, Sequence, Iterable, Tuple
import re

pio.renderers.default = "browser"
try:
    pio.kaleido.scope.mathjax = None  # Disable MathJax in static export
except Exception:
    pass  # Kaleido not available during import; handled at write_image time


_PERT_ORDER: Sequence[str] = [
    "Drift", "Attenuation", "Noise", "StuckSensor", "MissingData",
    "FasterSampling", "SlowerSampling", "Spike",
    "WrongState", "Chattering",
]


def _pretty(name: str) -> str:
    return "".join(w.capitalize() for w in name.replace("-", "_").split("_"))


def _normalize_scenario_name(name: str) -> str:
    if name is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def plot_scenario_radar(
    traces: Mapping[str, Mapping[str, float]] | Sequence[Tuple[str, Mapping[str, float]]],
    *,
    title: str | None = None,
    scenario_order: Sequence[str] | None = None,
    radial_range: Tuple[float, float] | None = (0.0, 1.0),
    fill_opacity: float = 0.25,
    color_sequence: Sequence[str] | None = None,
) -> go.Figure:
    """Render a spider/radar chart of scenario robustness scores.

    Args:
        traces: Mapping or sequence of (label, scenario->value) pairs. Values outside the provided scenarios are ignored. Missing scenarios are skipped.
        title: Optional plot title.
        scenario_order: Order of scenarios around the radar. Defaults to perturbation order.
        radial_range: (min, max) range for the radial axis.
        fill_opacity: alpha for the polygon fill.
    """
    if isinstance(traces, Mapping):
        items: Iterable[Tuple[str, Mapping[str, float]]] = traces.items()
    else:
        items = traces

    if scenario_order is None:
        scenario_order = [name for name in _PERT_ORDER]
    normalized_order = [_normalize_scenario_name(name) for name in scenario_order]

    def _order_by_normalized(names: Iterable[str]) -> list[str]:
        norm_map: dict[str, str] = {}
        for name in names:
            norm = _normalize_scenario_name(name)
            if norm not in norm_map:
                norm_map[norm] = name
        ordered: list[str] = []
        for norm in normalized_order:
            if norm in norm_map:
                ordered.append(norm_map.pop(norm))
        ordered.extend(name for _, name in sorted(norm_map.items()))
        return ordered

    # Collect all scenarios present in traces to avoid empty axes.
    present = set()
    normalized: list[Tuple[str, list[str], list[float]]] = []
    for label, values in items:
        available = {scenario: float(val) for scenario, val in values.items() if val is not None}
        if not available:
            continue
        present.update(available.keys())
        ordered_theta = _order_by_normalized(available.keys())
        ordered_r = [available[s] for s in ordered_theta]
        normalized.append((label, ordered_theta, ordered_r))

    if not normalized:
        fig = go.Figure()
        fig.update_layout(title=title or "Scenario Robustness (No Data)")
        return fig

    # Restrict theta to scenarios that appear somewhere.
    keep_theta = _order_by_normalized(present)

    fig = go.Figure()
    palette = list(color_sequence) if color_sequence is not None else px.colors.qualitative.Plotly
    fill_alpha = max(0.0, min(1.0, float(fill_opacity)))
    for idx, (label, theta_vals, r_vals) in enumerate(normalized):
        theta_filtered = [s for s in keep_theta if s in theta_vals]
        if not theta_filtered:
            continue
        value_map = dict(zip(theta_vals, r_vals))
        ordered_r = [value_map[s] for s in theta_filtered]
        # Close the polygon by repeating the first point.
        display_theta = [_pretty(s.replace(" ", "_")) for s in theta_filtered]
        closed_theta = display_theta + [display_theta[0]]
        closed_r = ordered_r + [ordered_r[0]]
        base_color = palette[idx % len(palette)] if palette else None
        fillcolor = None
        if base_color:
            if base_color.startswith("#") and len(base_color) == 7:
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                fillcolor = f"rgba({r},{g},{b},{fill_alpha})"
            elif base_color.startswith("rgb"):
                components = base_color.rstrip(")").split("(")[1]
                parts = components.split(",")
                if len(parts) >= 3:
                    r, g, b = parts[:3]
                    fillcolor = f"rgba({r.strip()},{g.strip()},{b.strip()},{fill_alpha})"
        if fillcolor is None:
            fillcolor = f"rgba(31,119,180,{fill_alpha})"
        hover_template = f"{label} — score=%{{r:.4f}}<extra></extra>"
        fig.add_trace(
            go.Scatterpolar(
                r=closed_r,
                theta=closed_theta,
                mode="lines+markers",
                name=label,
                fill="toself",
                fillcolor=fillcolor,
                line=dict(color=base_color) if base_color else None,
                marker=dict(color=base_color) if base_color else None,
                opacity=1.0,
                hovertemplate=hover_template,
            )
        )

    if radial_range:
        r_min, r_max = radial_range
    else:
        all_vals = [val for _, _, values in normalized for val in values]
        if all_vals:
            buffer = 0.05 * (max(all_vals) - min(all_vals) or 1.0)
            r_min = max(0.0, min(all_vals) - buffer)
            r_max = min(1.0, max(all_vals) + buffer)
        else:
            r_min, r_max = 0.0, 1.0

    default_title = "Scenario Robustness"
    fig.update_layout(
        title=title or default_title,
        polar=dict(
            radialaxis=dict(range=[r_min, r_max], tickfont=dict(size=12), ticks="outside", showline=True),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=len(normalized) > 1,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=16, family="Serif"),
        title_font=dict(size=20),
    )
    return fig


def plot_quantile_forecast(
    time_index: Sequence[float] | Sequence[int],
    target: Sequence[float],
    prediction: Sequence[float],
    *,
    quantile_predictions: Mapping[float, Sequence[float]] | None = None,
    clean_input: Sequence[Sequence[float]] | Sequence[float] | None = None,
    perturbed_input: Sequence[Sequence[float]] | Sequence[float] | None = None,
    input_time_index: Sequence[float] | Sequence[int] | None = None,
    input_feature_names: Sequence[str] | None = None,
    target_feature_names: Sequence[str] | None = None,
    affected_feature_names: Sequence[str] | None = None,
    title: str | None = None,
    scenario: str | None = None,
    severity: float | None = None,
    target_label: str = 'Target',
    prediction_label: str = 'Prediction',
) -> go.Figure:
    def _ensure_2d(values):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _norm(name: str | None) -> str:
        return name.strip().casefold() if isinstance(name, str) and name.strip() else ""

    def _default_time(length: int) -> np.ndarray:
        return np.arange(1, length + 1, dtype=float)

    output_time = np.asarray(time_index, dtype=float)
    target_arr = _ensure_2d(target)
    prediction_arr = _ensure_2d(prediction)

    input_names = list(input_feature_names or [])
    clean_arr = _ensure_2d(clean_input) if clean_input is not None else None
    pert_arr = _ensure_2d(perturbed_input) if perturbed_input is not None else None
    clean_time = None
    if clean_arr is not None:
        clean_time = (
            np.asarray(input_time_index, dtype=float)
            if input_time_index is not None
            else _default_time(clean_arr.shape[0])
        )

    target_name_keys = {_norm(name) for name in (target_feature_names or []) if _norm(name)}
    target_name_lookup = {
        _norm(name): name
        for name in (target_feature_names or [])
        if isinstance(name, str) and name.strip()
    }
    affected_name_keys = {_norm(name) for name in (affected_feature_names or []) if _norm(name)}
    normalized_inputs = [_norm(name) for name in input_names]

    input_traces: list[go.Scatter] = []
    if clean_arr is not None:
        for idx in range(clean_arr.shape[1]):
            name = input_names[idx] if idx < len(input_names) and input_names[idx] else f"Input {idx + 1}"
            norm_name = normalized_inputs[idx] if idx < len(normalized_inputs) else ""
            legend_label = name if name else f"Input {idx + 1}"
            is_target_channel = norm_name in target_name_keys
            if is_target_channel:
                legend_label = target_name_lookup.get(norm_name, legend_label)
                line_style = dict(color='rgb(0, 0, 0)', width=2)
                show_legend = False
            else:
                line_style = dict(color='rgba(128, 128, 128, 0.55)', width=1)
                show_legend = True
            input_traces.append(
                go.Scatter(
                    x=clean_time,
                    y=clean_arr[:, idx],
                    mode='lines',
                    name=legend_label,
                    line=line_style,
                    legendgroup='inputs',
                    legendrank=50 + idx,
                    showlegend=show_legend,
                )
            )

    perturbed_traces = []
    if pert_arr is not None:
        if clean_time is None:
            clean_time = (
                np.asarray(input_time_index, dtype=float)
                if input_time_index is not None
                else _default_time(pert_arr.shape[0])
            )
        affected_indices = {idx for idx, key in enumerate(normalized_inputs) if key in affected_name_keys}
        diff_indices = set()
        if clean_arr is not None and clean_arr.shape == pert_arr.shape:
            diff_mask = np.any(np.abs(pert_arr - clean_arr) > 1e-8, axis=0)
            diff_indices = {int(i) for i, flagged in enumerate(diff_mask) if flagged}
        candidate_indices = sorted(affected_indices or diff_indices)
        for pos, idx in enumerate(candidate_indices):
            if idx >= pert_arr.shape[1]:
                continue
            name = input_names[idx] if idx < len(input_names) and input_names[idx] else f"Input {idx + 1}"
            perturbed_traces.append(
                go.Scatter(
                    x=clean_time,
                    y=pert_arr[:, idx],
                    mode='lines',
                    name=f"{name if name else f'Input {idx + 1}'} (Perturbed)",
                    line=dict(color='rgba(214, 39, 40, 0.9)', width=2, dash='dashdot'),
                    legendgroup='perturbed',
                    legendrank=40 + pos,
                    showlegend=True,
                )
            )

    quantile_traces: list[go.Scatter] = []
    quantile_map: dict[float, np.ndarray] = {}
    if quantile_predictions:
        for level, values in quantile_predictions.items():
            level_float = float(level)
            quantile_map[level_float] = _ensure_2d(values)[:, 0]

        lower_levels = sorted(lvl for lvl in quantile_map if lvl < 0.5)
        upper_levels = sorted((lvl for lvl in quantile_map if lvl > 0.5), reverse=True)
        palette = px.colors.sequential.Blues
        for band_idx, (low, high) in enumerate(zip(lower_levels, upper_levels)):
            color = palette[min(band_idx * 2 + 2, len(palette) - 1)]
            quantile_traces.append(
                go.Scatter(
                    x=output_time,
                    y=quantile_map[high],
                    mode='lines',
                    line=dict(width=0),
                    hoverinfo='skip',
                    showlegend=False,
                    legendrank=30 + band_idx * 2,
                )
            )
            quantile_traces.append(
                go.Scatter(
                    x=output_time,
                    y=quantile_map[low],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=color,
                    name=f"Q{low:g}-Q{high:g}",
                    hoverinfo='skip',
                    legendrank=30 + band_idx * 2 + 1,
                )
            )

        if 0.5 in quantile_map:
            quantile_traces.append(
                go.Scatter(
                    x=output_time,
                    y=quantile_map[0.5],
                    mode='lines',
                    name='Median',
                    line=dict(color='rgba(44, 160, 44, 0.9)', width=2, dash='dash'),
                    legendrank=30 + len(quantile_traces),
                )
            )

    target_names = [name for name in (target_feature_names or []) if isinstance(name, str) and name]
    primary_target_name = target_names[0] if target_names else target_label
    target_primary = target_arr[:, 0]
    target_traces = [
        go.Scatter(
            x=output_time,
            y=target_primary,
            mode='lines',
            name=primary_target_name,
            line=dict(color='rgb(0, 0, 0)', width=2),
            legendrank=10,
        )
    ]
    for idx in range(1, target_arr.shape[1]):
        name = target_names[idx] if idx < len(target_names) else f"{target_label} {idx + 1}"
        target_traces.append(
            go.Scatter(
                x=output_time,
                y=target_arr[:, idx],
                mode='lines',
                name=name,
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1, dash='dot'),
                legendrank=11 + idx,
            )
        )

    primary_prediction_name = f"{primary_target_name} (Pred)" if target_names else prediction_label
    prediction_primary = prediction_arr[:, 0]
    prediction_traces: list[go.Scatter] = [
        go.Scatter(
            x=output_time,
            y=prediction_primary,
            mode='lines',
            name=primary_prediction_name,
            line=dict(color='rgb(31,119,180)', width=2),
            legendrank=20,
        )
    ]
    for idx in range(1, prediction_arr.shape[1]):
        name = target_names[idx] if idx < len(target_names) else f"{prediction_label} {idx + 1}"
        prediction_traces.append(
            go.Scatter(
                x=output_time,
                y=prediction_arr[:, idx],
                mode='lines',
                name=f"{name} (Pred)",
                line=dict(color='rgba(34, 139, 34, 0.3)', width=1, dash='dot'),
                legendrank=21 + idx,
            )
        )

    if scenario is None:
        raise ValueError('plot_quantile_forecast requires a perturbation scenario name.')
    if severity is None:
        raise ValueError('plot_quantile_forecast requires a severity value.')
    severity_value = float(severity)
    if not np.isfinite(severity_value):
        raise ValueError(f'Severity value must be finite, received {severity_value}.')

    fig = go.Figure()
    for trace in input_traces:
        fig.add_trace(trace)
    for trace in quantile_traces:
        fig.add_trace(trace)
    for trace in target_traces:
        fig.add_trace(trace)
    for trace in prediction_traces:
        fig.add_trace(trace)
    for trace in perturbed_traces:
        fig.add_trace(trace)

    default_title = f"Forecast vs Target | Scenario: {scenario}, Severity: {severity_value:.3f}"

    if output_time.size:
        fig.add_vline(
            x=output_time[0] - 0.5,
            line=dict(color='rgba(0,0,0,0.35)', dash='dash'),
            annotation_text='Forecast start',
            annotation_position='top left',
        )

    fig.update_layout(
        title=title or default_title,
        template='plotly_white',
        autosize=True,
        height=400,
        font=dict(size=16, family='Serif'),
        title_font=dict(size=20),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(traceorder='normal'),
    )
    fig.update_xaxes(
        title_text='Time',
        showgrid=True,
        gridcolor='rgba(0,0,0,0.15)',
        ticks='outside',
        ticklen=6,
        tickwidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        title_text='Value',
        showgrid=True,
        gridcolor='rgba(0,0,0,0.15)',
        ticks='outside',
        ticklen=6,
        tickwidth=1,
        mirror=True,
    )
    return fig


def plot_split_drift_bar(
    profile_df: pd.DataFrame,
    *,
    focus: str = "train_test",
    top_k: int = 12,
    height: int = 320,
) -> go.Figure:
    if not isinstance(profile_df, pd.DataFrame):
        profile_df = pd.DataFrame(profile_df)
    if profile_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No distribution profile available", template="plotly_white")
        return fig

    valid_focus = {"train_test", "val_test", "both"}
    if focus not in valid_focus:
        raise ValueError(f"Unsupported focus '{focus}', expected one of {sorted(valid_focus)}.")

    primary_col = "score_train_test" if focus != "val_test" else "score_val_test"
    df = profile_df.dropna(subset=[primary_col]).copy()
    if df.empty:
        df = profile_df.copy()
    df = df.sort_values(primary_col, ascending=False).head(top_k)

    def _summary(series: pd.Series) -> pd.Series:
        return series.fillna("n/a")

    traces_added = False
    fig = go.Figure()
    show_val = df["score_val_test"].notna().any()

    if focus in ("train_test", "both"):
        customdata = np.stack(
            [
                df["score_metric"],
                _summary(df["train_summary"]),
                _summary(df["test_summary"]),
                df["train_count"].fillna(0),
                df["test_count"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Bar(
                x=df["score_train_test"],
                y=df["feature"],
                name="Train vs Test",
                orientation="h",
                marker_color="#5b6af0",
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Comparison: Train vs Test<br>"
                    "Metric: %{customdata[0]}<br>"
                    "Score: %{x:.3f}<br>"
                    "Train: %{customdata[1]} (%{customdata[3]:.0f} rows)<br>"
                    "Test: %{customdata[2]} (%{customdata[4]:.0f} rows)<extra></extra>"
                ),
            )
        )
        traces_added = True

    if focus in ("val_test", "both") and show_val:
        customdata = np.stack(
            [
                df["score_metric"],
                _summary(df["val_summary"]),
                _summary(df["test_summary"]),
                df["val_count"].fillna(0),
                df["test_count"].fillna(0),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Bar(
                x=df["score_val_test"],
                y=df["feature"],
                name="Val vs Test",
                orientation="h",
                marker_color="#ffa94d",
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Comparison: Val vs Test<br>"
                    "Metric: %{customdata[0]}<br>"
                    "Score: %{x:.3f}<br>"
                    "Val: %{customdata[1]} (%{customdata[3]:.0f} rows)<br>"
                    "Test: %{customdata[2]} (%{customdata[4]:.0f} rows)<extra></extra>"
                ),
            )
        )
        traces_added = True

    if not traces_added:
        fig.add_trace(
            go.Bar(
                x=df[primary_col],
                y=df["feature"],
                orientation="h",
                marker_color="#5b6af0",
                name="Drift",
            )
        )

    for threshold in (0.1, 0.2, 0.3):
        fig.add_vline(
            x=threshold,
            line_dash="dot",
            line_color="rgba(70,70,70,0.35)",
        )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        height=height,
        margin=dict(l=10, r=10, t=12, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis=dict(title="Drift score"),
        yaxis=dict(title="", automargin=True, autorange="reversed"),
    )
    return fig


def plot_feature_kde(
    feature_x: str,
    feature_y: str,
    kde_data: Mapping[str, Mapping[str, Sequence[float]]],
    visible_scenarios: Sequence[str] | None = None,
    odd_clip_quantiles: tuple[float, float] | None = None,
    odd_kde_mass: float = 0.98,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    if not feature_x or not feature_y or not kde_data:
        fig.update_layout(template="plotly_white", height=height)
        return fig

    def _order_scenarios(keys: Sequence[str]) -> list[str]:
        priorities = ["train", "val", "test"] + list(_PERT_ORDER)
        norm_priorities = [_normalize_scenario_name(p) for p in priorities]
        norm_map: dict[str, str] = {}
        for key in keys:
            norm = _normalize_scenario_name(key)
            if norm not in norm_map:
                norm_map[norm] = key
        ordered: list[str] = []
        for norm in norm_priorities:
            if norm in norm_map:
                ordered.append(norm_map.pop(norm))
        ordered.extend(name for _, name in sorted(norm_map.items()))
        return ordered

    scenario_colors = {
        "train": "#5b6af0",
        "val": "#4dabf7",
        "test": "#ffa94d",
    }
    scenarios = _order_scenarios([s for s in kde_data.keys() if s != "odd"])
    # IndPenSim visualization: exclude other perturbation scenarios to focus on Drift KDE.
    allowed_norms = {"train", "val", "test", "drift"}
    scenarios = [sc for sc in scenarios if _normalize_scenario_name(sc) in allowed_norms]
    extra_palette = px.colors.qualitative.Safe + px.colors.qualitative.Dark24 + px.colors.qualitative.Pastel
    extra_idx = 0
    for sc in scenarios:
        if sc in scenario_colors:
            continue
        scenario_colors[sc] = extra_palette[extra_idx % len(extra_palette)]
        extra_idx += 1
    visible_set = set(visible_scenarios) if visible_scenarios else None

    def _convex_hull_coords(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.column_stack([x_vals, y_vals])
        hull = ConvexHull(pts)
        coords = pts[hull.vertices]
        coords = np.vstack([coords, coords[0]])
        return coords[:, 0], coords[:, 1]

    def _finite(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    def _apply_odd_clip(values: Sequence[float]) -> np.ndarray:
        arr = _finite(values)
        if arr.size == 0 or odd_clip_quantiles is None:
            return arr
        q_low, q_high = odd_clip_quantiles
        q_vals = np.quantile(arr, [float(q_low), float(q_high)])
        return arr[(arr >= q_vals[0]) & (arr <= q_vals[1])]

    def _apply_odd_clip_paired(x_values: Sequence[float], y_values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        min_len = min(x_arr.size, y_arr.size)
        if min_len == 0:
            return np.asarray([]), np.asarray([])
        x_arr = x_arr[:min_len]
        y_arr = y_arr[:min_len]
        pair_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[pair_mask]
        y_arr = y_arr[pair_mask]
        if x_arr.size == 0 or y_arr.size == 0:
            return np.asarray([]), np.asarray([])
        if odd_clip_quantiles is None:
            return x_arr, y_arr
        q_low, q_high = odd_clip_quantiles
        x_q = np.quantile(x_arr, [float(q_low), float(q_high)])
        y_q = np.quantile(y_arr, [float(q_low), float(q_high)])
        clip_mask = (
            (x_arr >= x_q[0])
            & (x_arr <= x_q[1])
            & (y_arr >= y_q[0])
            & (y_arr <= y_q[1])
        )
        return x_arr[clip_mask], y_arr[clip_mask]

    def _density_threshold(density_values: np.ndarray, mass_level: float) -> float:
        """Return density cutoff whose superlevel set covers ~mass_level probability mass."""
        vals = np.asarray(density_values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan")
        sorted_vals = np.sort(vals)[::-1]
        cdf = np.cumsum(sorted_vals)
        cdf /= cdf[-1]
        idx = np.searchsorted(cdf, mass_level)
        idx = min(idx, sorted_vals.size - 1)
        return float(sorted_vals[idx])

    def _odd_band_1d(train_vals: np.ndarray, mass_level: float) -> tuple[float, float] | None:
        if train_vals.size < 3 or np.std(train_vals) == 0:
            return None
        kde = gaussian_kde(train_vals)
        x_grid = np.linspace(np.min(train_vals), np.max(train_vals), 512)
        density = kde(x_grid)
        cutoff = _density_threshold(density, mass_level)
        if not np.isfinite(cutoff):
            return None
        mask = density >= cutoff
        if not np.any(mask):
            return None
        return float(np.min(x_grid[mask])), float(np.max(x_grid[mask]))

    def _odd_hull_2d(train_x: np.ndarray, train_y: np.ndarray, mass_level: float) -> tuple[np.ndarray, np.ndarray] | None:
        if train_x.size < 3 or train_y.size < 3:
            return None
        if np.std(train_x) == 0 or np.std(train_y) == 0:
            return None
        kde = gaussian_kde(np.vstack([train_x, train_y]))
        x_grid = np.linspace(np.min(train_x), np.max(train_x), 70)
        y_grid = np.linspace(np.min(train_y), np.max(train_y), 70)
        xv, yv = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([xv.ravel(), yv.ravel()])
        z = kde(positions).reshape(xv.shape)
        cutoff = _density_threshold(z.ravel(), mass_level)
        if not np.isfinite(cutoff):
            return None
        region_mask = z >= cutoff
        if not np.any(region_mask):
            return None
        region_pts = np.column_stack([xv[region_mask], yv[region_mask]])
        region_pts = region_pts[np.isfinite(region_pts).all(axis=1)]
        if region_pts.shape[0] < 3:
            return None
        return _convex_hull_coords(region_pts[:, 0], region_pts[:, 1])

    train_split = kde_data.get("train", {})
    train_x_1d = _apply_odd_clip(train_split.get(feature_x, [])) if isinstance(train_split, dict) else np.asarray([])
    train_x, train_y = (
        _apply_odd_clip_paired(train_split.get(feature_x, []), train_split.get(feature_y, []))
        if isinstance(train_split, dict)
        else (np.asarray([]), np.asarray([]))
    )
    odd_mass_level = float(odd_kde_mass)
    if not (0.0 < odd_mass_level < 1.0):
        odd_mass_level = 0.98

    if feature_x == feature_y:
        for sc in scenarios:
            vals_raw = kde_data.get(sc, {}).get(feature_x)
            if vals_raw is None:
                continue
            x_values = _finite(vals_raw)
            if x_values.size < 2 or np.std(x_values) == 0:
                continue
            kde = gaussian_kde(x_values)
            x_grid = np.linspace(np.min(x_values), np.max(x_values), 200)
            density = kde(x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=density,
                    mode="lines",
                    line=dict(color=scenario_colors.get(sc, "#5b6af0"), width=2),
                    fill="tozeroy",
                    name=_pretty(sc),
                    legendgroup=sc,
                    hoverinfo="skip",
                    visible=True if (visible_set is None or sc in visible_set) else "legendonly",
                )
            )
        odd_band = _odd_band_1d(train_x_1d, odd_mass_level)
        if odd_band:
            odd_min, odd_max = odd_band
            fig.add_vrect(
                x0=odd_min,
                x1=odd_max,
                fillcolor="rgba(128,0,128,0.15)",
                line=dict(color="rgba(128,0,128,0.35)", width=1),
                layer="below",
                annotation_text="ODD",
                annotation_position="top left",
                annotation_font_color="rgba(80,0,80,0.6)",
            )
        y_title = "Density"
    else:
        odd_hull = _odd_hull_2d(train_x, train_y, odd_mass_level)
        pairs = []
        for sc in scenarios:
            x_vals = _finite(kde_data.get(sc, {}).get(feature_x, []))
            y_vals = _finite(kde_data.get(sc, {}).get(feature_y, []))
            if x_vals.size < 2 or y_vals.size < 2:
                continue
            pairs.append((x_vals, y_vals))
        if not pairs:
            fig.update_layout(template="plotly_white", height=height)
            return fig
        all_x = np.concatenate([p[0] for p in pairs])
        all_y = np.concatenate([p[1] for p in pairs])
        if not np.isfinite(all_x).any() or not np.isfinite(all_y).any():
            fig.update_layout(template="plotly_white", height=height)
            return fig
        x_grid = np.linspace(np.min(all_x), np.max(all_x), 70)
        y_grid = np.linspace(np.min(all_y), np.max(all_y), 70)
        xv, yv = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([xv.ravel(), yv.ravel()])

        # Pre-add ODD hull so its legend entry precedes scenario traces.
        if odd_hull is None:
            odd_pts = np.column_stack([train_x, train_y])
            odd_pts = odd_pts[np.isfinite(odd_pts).all(axis=1)]
            if odd_pts.shape[0] >= 3:
                odd_hull = _convex_hull_coords(odd_pts[:, 0], odd_pts[:, 1])
        if odd_hull is not None:
            hull_x, hull_y = odd_hull
            fig.add_trace(
                go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode="lines",
                    line=dict(color="rgba(128,0,128,0.9)", width=2, dash="dash"),
                    fill="toself",
                    fillcolor="rgba(128,0,128,0.15)",
                    name="ODD",
                    hoverinfo="skip",
                )
            )

        for sc in scenarios:
            x_values = _finite(kde_data.get(sc, {}).get(feature_x, []))
            y_values = _finite(kde_data.get(sc, {}).get(feature_y, []))
            if x_values.size < 3 or y_values.size < 3:
                continue
            if np.std(x_values) == 0 or np.std(y_values) == 0:
                continue
            kde = gaussian_kde(np.vstack([x_values, y_values]))
            z = kde(positions).reshape(xv.shape)
            color = scenario_colors.get(sc, "#5b6af0")
            fig.add_trace(
                go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=z,
                    ncontours=5,
                    autocolorscale=False,
                    colorscale=[[0, color], [1, color]],
                    line=dict(color=color, width=2),
                    contours=dict(coloring="lines", showlabels=False),
                    showscale=False,
                    name=_pretty(sc),
                    opacity=0.9,
                    legendgroup=sc,
                    showlegend=True,
                    visible=True if (visible_set is None or sc in visible_set) else "legendonly",
                    hoverinfo="skip",
                )
            )
        y_title = feature_y

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=12, b=18),
        height=height,
        xaxis=dict(title=feature_x, zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title=y_title, zeroline=False, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        showlegend=True,
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
