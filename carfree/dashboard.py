"""Scenario tradeoffs dashboard for stakeholders.

Runs the scenario suite plus a bulk-discount sweep and renders a six-panel
PNG (and a self-contained HTML report) comparing ridership, revenue,
service frequency, convenience, pass adoption, and the ridership-lift vs
revenue tradeoff curve.

Colors follow a validated categorical palette (fixed slot order, one hue
per scenario everywhere it appears); observed data is always neutral gray.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .engine import simulate
from .params import ModelParams, SCENARIOS
from .results import SimulationResult

# validated categorical palette (light mode) -- color follows the scenario
SCENARIO_COLORS = {
    "status_quo": "#2a78d6",       # blue
    "car_free": "#eb6834",         # orange
    "car_free_passes": "#1baf7a",  # aqua
    "passes_only": "#eda100",      # yellow (low contrast -> direct labels)
}
SCENARIO_LABELS = {
    "status_quo": "Status quo",
    "car_free": "Car-free downtown",
    "car_free_passes": "Car-free + employer passes",
    "passes_only": "Employer passes only",
}
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def run_suite(params: ModelParams, n_agents: int = 20_000, years: int = 5,
              seed: int = 7, verbose: bool = True) -> Dict[str, SimulationResult]:
    results = {}
    for name in SCENARIOS:
        results[name] = simulate(name, params=params, n_agents=n_agents,
                                 n_days=365 * years, seed=seed)
        if verbose:
            kpi = results[name].kpis()
            print(f"  {name:<18} annual boardings {kpi['annual_boardings']:>13,.0f}  "
                  f"annual revenue ${kpi['annual_total_revenue']:>12,.0f}  "
                  f"({results[name].meta['runtime_s']:.1f}s)")
    return results


def sweep_bulk_discount(params: ModelParams,
                        discounts: Sequence[float] = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8),
                        n_agents: int = 20_000, years: int = 5, seed: int = 7,
                        reference: Optional[SimulationResult] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """Ridership-lift vs revenue tradeoff of the pass program's bulk discount,
    measured against the car-free scenario without an active program."""
    ref = reference or simulate("car_free", params=params, n_agents=n_agents,
                                n_days=365 * years, seed=seed)
    ref_kpi = ref.kpis()
    rows: List[dict] = []
    for d in discounts:
        res = simulate("car_free_passes",
                       params=params.with_updates(bulk_discount=d),
                       n_agents=n_agents, n_days=365 * years, seed=seed)
        kpi = res.kpis()
        rows.append({
            "bulk_discount": d,
            "annual_boardings": kpi["annual_boardings"],
            "annual_total_revenue": kpi["annual_total_revenue"],
            "ridership_lift_pct": 100 * (kpi["annual_boardings"]
                                         / ref_kpi["annual_boardings"] - 1),
            "revenue_delta_pct": 100 * (kpi["annual_total_revenue"]
                                        / ref_kpi["annual_total_revenue"] - 1),
            "pass_holder_share": kpi["pass_holder_share"],
            "employer_offer_share": kpi["employer_offer_share"],
            "revenue_per_boarding": kpi["revenue_per_boarding"],
        })
        if verbose:
            r = rows[-1]
            print(f"  discount {d:.2f}: lift {r['ridership_lift_pct']:+5.1f}%  "
                  f"revenue {r['revenue_delta_pct']:+5.1f}%  "
                  f"pass share {r['pass_holder_share']:.1%}")
    return pd.DataFrame(rows)


def kpi_table(results: Dict[str, SimulationResult]) -> pd.DataFrame:
    return pd.DataFrame({SCENARIO_LABELS[k]: v.kpis() for k, v in results.items()}).T


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _style_axis(ax, title: str, ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8,
                 fontweight="semibold")
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.margins(x=0.01)


def _monthly(res: SimulationResult) -> pd.DataFrame:
    m = res.monthly
    m = m[m["days"] >= 28]                      # drop partial edge months
    m = m.set_axis(m.index.to_timestamp(), axis=0)
    return m


def _policy_line(ax, results: Dict[str, SimulationResult]):
    res = results.get("car_free")
    if res is None or res.scenario.policy_start_day is None:
        return
    t = res.daily.index[0] + pd.Timedelta(days=res.scenario.policy_start_day)
    ax.axvline(t, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("car ban", xy=(t, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -2), textcoords="offset points",
                fontsize=7.5, color=MUTED, va="top")


def _scenario_lines(ax, results, column, scale=1.0, monthly_frames=None):
    for name, res in results.items():
        m = monthly_frames[name] if monthly_frames else _monthly(res)
        ax.plot(m.index, m[column] * scale, color=SCENARIO_COLORS[name],
                linewidth=1.8, label=SCENARIO_LABELS[name],
                solid_capstyle="round")


def make_dashboard(results: Dict[str, SimulationResult], sweep: pd.DataFrame,
                   out_png: str = "outputs/dashboard.png",
                   out_html: Optional[str] = "outputs/dashboard.html",
                   validation: Optional[dict] = None) -> str:
    frames = {name: _monthly(res) for name, res in results.items()}

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), dpi=150)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.09,
                        wspace=0.24, hspace=0.42)
    fig.suptitle("Baltimore car-free city: bus ridership scenario tradeoffs",
                 fontsize=15, color=INK, x=0.055, ha="left", y=0.975,
                 fontweight="bold")
    fig.text(0.055, 0.925,
             "Agent-based simulation, 5-year horizon - calibrated to MDOT MTA "
             "monthly bus ridership (2019)", fontsize=9.5, color=INK2)

    ax = axes[0, 0]
    _scenario_lines(ax, results, "boardings", 1e-6, frames)
    _style_axis(ax, "Monthly bus boardings", "millions")
    _policy_line(ax, results)

    ax = axes[0, 1]
    _scenario_lines(ax, results, "total_revenue", 1e-6, frames)
    _style_axis(ax, "Monthly operator revenue (fares + pass sales)", "$ millions")
    _policy_line(ax, results)

    ax = axes[0, 2]
    _scenario_lines(ax, results, "frequency", 1.0, frames)
    _style_axis(ax, "Service frequency", "buses / hour (system avg)")
    _policy_line(ax, results)

    ax = axes[1, 0]
    _scenario_lines(ax, results, "convenience", 1.0, frames)
    _style_axis(ax, "Rider convenience index", "0-1 composite")
    _policy_line(ax, results)

    ax = axes[1, 1]
    for name in ("car_free_passes", "passes_only"):
        if name not in frames:
            continue
        m = frames[name]
        c = SCENARIO_COLORS[name]
        ax.plot(m.index, 100 * m["pass_rider_share"], color=c, linewidth=1.8,
                label=f"{SCENARIO_LABELS[name]} - riders w/ pass")
        ax.plot(m.index, 100 * m["offer_share"], color=c, linewidth=1.4,
                linestyle=(0, (4, 2)),
                label=f"{SCENARIO_LABELS[name]} - employers offering")
    _style_axis(ax, "Employer pass adoption", "% (solid: riders, dashed: employers)")
    _policy_line(ax, results)

    ax = axes[1, 2]
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.plot(sweep["ridership_lift_pct"], sweep["revenue_delta_pct"],
            color=SCENARIO_COLORS["car_free_passes"], linewidth=1.8, zorder=2)
    ax.scatter(sweep["ridership_lift_pct"], sweep["revenue_delta_pct"],
               s=42, color=SCENARIO_COLORS["car_free_passes"], zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    for _, r in sweep.iterrows():
        ax.annotate(f"{r['bulk_discount']:.0%}",
                    (r["ridership_lift_pct"], r["revenue_delta_pct"]),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=7.5, color=INK2)
    _style_axis(ax, "Pass-program tradeoff by bulk discount",
                "revenue vs car-free baseline, %")
    ax.set_xlabel("ridership lift vs car-free baseline, %",
                  fontsize=8.5, color=INK2)
    ax.margins(x=0.14, y=0.16)   # keep point labels inside the panel

    for ax in axes.flat[:5]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.985, 0.97),
               labelcolor=INK2)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=PAGE)
    plt.close(fig)

    if out_html:
        _write_html(out_html, out_png, results, sweep, validation)
    return out_png


def _write_html(out_html: str, png_path: str,
                results: Dict[str, SimulationResult], sweep: pd.DataFrame,
                validation: Optional[dict]) -> None:
    img64 = base64.b64encode(Path(png_path).read_bytes()).decode()
    kpis = kpi_table(results)
    fmt = {
        "annual_boardings": "{:,.0f}", "annual_fare_revenue": "${:,.0f}",
        "annual_pass_revenue": "${:,.0f}", "annual_total_revenue": "${:,.0f}",
        "mean_bus_share": "{:.1%}", "mean_convenience": "{:.2f}",
        "mean_wait_min": "{:.1f}", "final_frequency": "{:.1f}",
        "pass_holder_share": "{:.1%}", "employer_offer_share": "{:.1%}",
        "revenue_per_boarding": "${:.2f}",
    }
    body = kpis.copy()
    for col, f in fmt.items():
        body[col] = body[col].map(f.format)
    table = body.to_html(border=0)

    val_html = ""
    if validation:
        color = {"PASS": "#0ca30c", "WARN": "#fab219", "FAIL": "#d03b3b"}
        val_html = (
            f"<p>Baseline validation vs observed MTA ridership: "
            f"<strong style='color:{color[validation['status']]}'>"
            f"{validation['status']}</strong> "
            f"(MAPE {validation['mape']:.2%}, window "
            f"{'-'.join(validation['meta']['window'])})</p>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Baltimore car-free city - scenario dashboard</title>
<style>
 body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        background:{PAGE}; color:{INK}; margin: 2rem auto; max-width: 1200px;
        padding: 0 1rem; }}
 img {{ max-width: 100%; border: 1px solid {GRID}; border-radius: 6px; }}
 table {{ border-collapse: collapse; font-size: 0.85rem; margin: 1rem 0; }}
 th, td {{ text-align: right; padding: 6px 12px;
          border-bottom: 1px solid {GRID}; }}
 th:first-child, td:first-child {{ text-align: left; }}
 h1 {{ font-size: 1.4rem; }} p {{ color:{INK2}; }}
</style></head><body>
<h1>Baltimore car-free city &mdash; bus ridership scenario tradeoffs</h1>
<p>Final-year KPIs per scenario (agent-based simulation, 5-year horizon,
calibrated to MDOT MTA 2019 monthly ridership).</p>
{val_html}
{table}
<img src="data:image/png;base64,{img64}" alt="scenario dashboard">
</body></html>"""
    Path(out_html).write_text(html, encoding="utf-8")


def make_impact_chart(results: Dict[str, SimulationResult],
                      out_png: str = "outputs/impact.png") -> str:
    """Net-impact summary: final-year % change vs status quo on four KPIs,
    each oriented so that a bar right of zero is an improvement."""
    def final_year(res):
        d = res.daily.iloc[-365:]
        m = res.monthly.iloc[-12:]
        return {
            "boardings": d["boardings"].sum(),
            "revenue": m["total_revenue"].sum(),
            "convenience": d["convenience"].mean(),
            "car_trips": d["car_users"].sum(),
        }

    base = final_year(results["status_quo"])
    scenarios = [s for s in ("car_free", "car_free_passes", "passes_only")
                 if s in results]
    kpis = [                       # (label, key, orientation)
        ("Bus boardings", "boardings", +1),
        ("Operator revenue", "revenue", +1),
        ("Rider convenience", "convenience", +1),
        ("Car trips avoided", "car_trips", -1),   # fewer cars = positive
    ]
    deltas = {
        s: [sign * (final_year(results[s])[key] / base[key] - 1) * 100
            for _, key, sign in kpis]
        for s in scenarios
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=150)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.80, bottom=0.10)
    fig.suptitle("Is the policy a net win? Final-year change vs status quo",
                 fontsize=13, color=INK, x=0.16, ha="left", fontweight="bold")
    fig.text(0.16, 0.885, "Every KPI oriented so right of zero = improvement. "
             "Preliminary simulation results, not a forecast.",
             fontsize=9, color=INK2)

    n_s = len(scenarios)
    bar_h = 0.72 / n_s
    for j, s in enumerate(scenarios):
        y = [i + (j - (n_s - 1) / 2) * bar_h for i in range(len(kpis))]
        bars = ax.barh(y, deltas[s], height=bar_h * 0.9,
                       color=SCENARIO_COLORS[s], label=SCENARIO_LABELS[s])
        for rect, v in zip(bars, deltas[s]):
            ax.annotate(f"{v:+.0f}%",
                        (v, rect.get_y() + rect.get_height() / 2),
                        xytext=(5 if v >= 0 else -5, 0),
                        textcoords="offset points", va="center",
                        ha="left" if v >= 0 else "right",
                        fontsize=8, color=INK2)
    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_yticks(range(len(kpis)))
    ax.set_yticklabels([k[0] for k in kpis], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_xlabel("% change vs status quo (final simulated year)",
                  fontsize=8.5, color=INK2)
    ax.margins(x=0.14)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2,
              loc="lower right")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=PAGE)
    plt.close(fig)
    return out_png


def make_benchmark_chart(rows: List[dict],
                         out_png: str = "outputs/benchmark.png") -> str:
    df = pd.DataFrame(rows)
    naive = df[df.engine == "naive"].reset_index(drop=True)
    fast = df[df.engine == "vectorized"].reset_index(drop=True)
    x = range(len(naive))
    colors = {"naive": "#eb6834", "vectorized": "#2a78d6"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.14, wspace=0.22)
    fig.suptitle("Per-agent loop vs vectorized engine (identical random streams)",
                 fontsize=12.5, color=INK, x=0.07, ha="left", fontweight="bold")

    for ax, col, title, unit in ((axes[0], "runtime_s", "Runtime", "seconds"),
                                 (axes[1], "peak_mem_mb", "Peak memory", "MB")):
        w = 0.36
        b1 = ax.bar([i - w / 2 for i in x], naive[col], width=w,
                    color=colors["naive"], label="naive per-agent loop")
        b2 = ax.bar([i + w / 2 for i in x], fast[col], width=w,
                    color=colors["vectorized"], label="vectorized")
        ax.set_yscale("log")
        for bars in (b1, b2):
            for rect in bars:
                v = rect.get_height()
                ax.annotate(f"{v:,.2f}" if v < 10 else f"{v:,.0f}",
                            (rect.get_x() + rect.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7.5, color=INK2,
                            xytext=(0, 2), textcoords="offset points")
        ratios = (naive[col] / fast[col]).round(0)
        _style_axis(ax, f"{title} ({unit}, log scale)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f"{n:,} agents\n({int(r)}x)" for n, r in zip(naive.n_agents, ratios)],
            fontsize=8, color=INK2)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.8)
        ax.grid(False, axis="y", which="minor")
        ax.tick_params(which="minor", length=0)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=PAGE)
    plt.close(fig)
    return out_png


def make_calibration_chart(validation: dict,
                           out_png: str = "outputs/calibration_fit.png") -> str:
    months = list(validation["monthly"].keys())
    obs = [v["observed"] / 1e6 for v in validation["monthly"].values()]
    sim = [v["simulated"] / 1e6 for v in validation["monthly"].values()]
    x = range(len(months))

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16)
    ax.plot(x, obs, color=INK2, linewidth=1.8, linestyle=(0, (4, 2)),
            label="Observed (MDOT MTA)")
    ax.plot(x, sim, color="#2a78d6", linewidth=1.8, label="Simulated baseline")
    _style_axis(ax, "Calibrated baseline vs observed monthly bus boardings",
                "boardings, millions")
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, fontsize=7.5, color=MUTED, rotation=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    ax.set_ylim(bottom=0)
    ax.text(0.99, 0.92, f"MAPE {validation['mape']:.2%} - status "
            f"{validation['status']}", fontsize=9, color=INK2,
            transform=ax.transAxes, ha="right")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=PAGE)
    plt.close(fig)
    return out_png
