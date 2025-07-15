"""Calibration against observed Maryland MTA monthly bus ridership.

Three things are fitted, in order:

  1. seasonal_factors -- month-of-year demand multipliers, extracted from the
     observed series (daily-mean normalized so month length doesn't leak in);
  2. asc_bus          -- the bus alternative-specific constant, tuned by
     bisection until the simulated bus mode share hits a plausible target;
  3. persons_per_agent -- the level scale mapping agent boardings to
     real-world boardings, closed-form from the ratio of totals.

The default calibration window is calendar 2019 (the last structurally
stable pre-COVID year in the observed series). Results are persisted to
outputs/calibration.json and consumed by `carfree validate` and the
dashboard.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .engine import simulate
from .params import ModelParams

DEFAULT_DATA = "datasets/ridership_data.csv"
DEFAULT_CALIBRATION = "outputs/calibration.json"
DEFAULT_WINDOW = ("2019-01", "2019-12")


def load_observed(path: str = DEFAULT_DATA) -> pd.Series:
    """Monthly observed boardings as a Series with a monthly PeriodIndex."""
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(pd.to_datetime(df["Date"], format="%b %Y"), freq="M")
    return pd.Series(df["Ridership"].to_numpy(dtype=float), index=idx, name="observed")


def seasonal_factors_from_observed(observed: pd.Series,
                                   window: Tuple[str, str] = DEFAULT_WINDOW,
                                   ) -> Tuple[float, ...]:
    """Month-of-year multipliers with mean 1.0, from daily-mean ridership."""
    obs = observed[pd.Period(window[0], "M"): pd.Period(window[1], "M")]
    daily_mean = obs / obs.index.days_in_month
    by_month = daily_mean.groupby(daily_mean.index.month).mean()
    factors = by_month / by_month.mean()
    return tuple(float(factors.get(m, 1.0)) for m in range(1, 13))


def _window_run(params: ModelParams, n_agents: int, seed: int,
                window: Tuple[str, str]):
    start = pd.Period(window[0], "M").to_timestamp()
    end = pd.Period(window[1], "M").to_timestamp(how="end")
    n_days = (end.normalize() - start).days + 1
    return simulate("status_quo", params=params, n_agents=n_agents,
                    n_days=n_days, start=str(start.date()), seed=seed)


def calibrate(
    data_path: str = DEFAULT_DATA,
    n_agents: int = 20_000,
    n_fit_agents: int = 6_000,
    seed: int = 7,
    window: Tuple[str, str] = DEFAULT_WINDOW,
    target_bus_share: float = 0.20,
    asc_bounds: Tuple[float, float] = (-3.0, 0.5),
    out_path: Optional[str] = DEFAULT_CALIBRATION,
    verbose: bool = True,
) -> dict:
    observed = load_observed(data_path)
    seasonal = seasonal_factors_from_observed(observed, window)
    base = ModelParams(seasonal_factors=seasonal, persons_per_agent=1.0)

    def bus_share(asc: float) -> float:
        res = _window_run(base.with_updates(asc_bus=asc), n_fit_agents, seed, window)
        d = res.daily
        return float(d["bus_riders"].sum() / d["travelers"].sum())

    # bus share is monotonically increasing in asc_bus -> bisection
    lo, hi = asc_bounds
    for it in range(10):
        mid = 0.5 * (lo + hi)
        share = bus_share(mid)
        if verbose:
            print(f"  [fit {it + 1:2d}] asc_bus={mid:+.4f} -> bus share {share:.4f} "
                  f"(target {target_bus_share:.2f})")
        if share < target_bus_share:
            lo = mid
        else:
            hi = mid
    asc_fit = 0.5 * (lo + hi)

    # final run at full population; level scale is closed-form
    final = _window_run(base.with_updates(asc_bus=asc_fit), n_agents, seed, window)
    sim_monthly = final.monthly["boardings"]
    obs_window = observed[pd.Period(window[0], "M"): pd.Period(window[1], "M")]
    scale = float(obs_window.sum() / sim_monthly.sum())

    sim_scaled = sim_monthly * scale
    err = (sim_scaled - obs_window) / obs_window
    mape = float(err.abs().mean())

    calib = {
        "asc_bus": round(asc_fit, 6),
        "persons_per_agent": round(scale, 4),
        "seasonal_factors": [round(f, 6) for f in seasonal],
        "fit": {
            "bus_share": float(final.daily["bus_riders"].sum()
                               / final.daily["travelers"].sum()),
            "target_bus_share": target_bus_share,
            "mape": mape,
            "max_abs_month_error": float(err.abs().max()),
            "monthly": {
                str(k): {"observed": float(o), "simulated": float(s),
                         "pct_error": float(e)}
                for k, o, s, e in zip(obs_window.index, obs_window, sim_scaled, err)
            },
        },
        "meta": {
            "data_path": data_path,
            "window": list(window),
            "n_agents": n_agents,
            "seed": seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(calib, indent=2))
        if verbose:
            print(f"  calibration written to {out_path}")
    if verbose:
        print(f"  asc_bus={asc_fit:+.4f}  persons_per_agent={scale:.2f}  "
              f"MAPE={mape:.2%}")
    return calib


def load_calibration(path: str = DEFAULT_CALIBRATION) -> dict:
    return json.loads(Path(path).read_text())


def apply_calibration(params: ModelParams, calib: dict) -> ModelParams:
    return params.with_updates(
        asc_bus=calib["asc_bus"],
        persons_per_agent=calib["persons_per_agent"],
        seasonal_factors=tuple(calib["seasonal_factors"]),
    )


def calibrated_params(calib_path: str = DEFAULT_CALIBRATION, **fallback_kwargs) -> ModelParams:
    """Load calibrated params, or defaults if no calibration file exists."""
    p = Path(calib_path)
    if p.exists():
        return apply_calibration(ModelParams(**fallback_kwargs), load_calibration(calib_path))
    return ModelParams(**fallback_kwargs)
