"""Automated divergence detection: simulated vs observed ridership.

`run_validation` re-runs the calibrated baseline over an observation window
and compares monthly boardings against the observed MTA series with three
checks:

  * per-month deviation  -- any |error| above `month_warn` is flagged;
  * window MAPE          -- above `mape_fail` the run FAILs;
  * rolling drift        -- a 3-month rolling *signed* mean error above
                            `drift_warn` flags systematic bias that per-month
                            checks can miss.

Statuses: PASS (exit 0), WARN (exit 0, or 1 with strict=True), FAIL (exit 1).
A JSON report is written for CI artifacts / cron jobs, so this doubles as a
regression gate for model-code changes and as a drift monitor when new
observed months are appended to the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .calibration import (DEFAULT_CALIBRATION, DEFAULT_DATA, DEFAULT_WINDOW,
                          apply_calibration, load_calibration, load_observed,
                          _window_run)
from .params import ModelParams

DEFAULT_REPORT = "outputs/validation_report.json"


@dataclass(frozen=True)
class Thresholds:
    month_warn: float = 0.10   # |monthly error| that flags a month
    mape_fail: float = 0.08    # window MAPE that fails the run
    drift_warn: float = 0.06   # |rolling-3 signed mean error| that flags drift


def run_validation(
    calib_path: str = DEFAULT_CALIBRATION,
    data_path: str = DEFAULT_DATA,
    window: Tuple[str, str] = DEFAULT_WINDOW,
    thresholds: Thresholds = Thresholds(),
    n_agents: Optional[int] = None,
    seed: Optional[int] = None,
    report_path: Optional[str] = DEFAULT_REPORT,
    verbose: bool = True,
) -> dict:
    calib = load_calibration(calib_path)
    params = apply_calibration(ModelParams(), calib)
    n_agents = n_agents or calib["meta"]["n_agents"]
    seed = seed if seed is not None else calib["meta"]["seed"]

    observed = load_observed(data_path)
    obs = observed[pd.Period(window[0], "M"): pd.Period(window[1], "M")]
    if obs.empty:
        raise ValueError(f"No observed months in window {window} in {data_path}")

    result = _window_run(params, n_agents, seed, window)
    sim = result.monthly["boardings"].reindex(obs.index)

    err = (sim - obs) / obs
    mape = float(err.abs().mean())
    rolling = err.rolling(3, min_periods=3).mean()

    flagged_months = [str(k) for k, e in err.items() if abs(e) > thresholds.month_warn]
    drift_months = [str(k) for k, e in rolling.items()
                    if pd.notna(e) and abs(e) > thresholds.drift_warn]

    if mape > thresholds.mape_fail:
        status = "FAIL"
    elif flagged_months or drift_months:
        status = "WARN"
    else:
        status = "PASS"

    report = {
        "status": status,
        "mape": mape,
        "max_abs_month_error": float(err.abs().max()),
        "mean_signed_error": float(err.mean()),
        "flagged_months": flagged_months,
        "drift_months": drift_months,
        "thresholds": {
            "month_warn": thresholds.month_warn,
            "mape_fail": thresholds.mape_fail,
            "drift_warn": thresholds.drift_warn,
        },
        "monthly": {
            str(k): {
                "observed": float(o),
                "simulated": float(s),
                "pct_error": float(e),
                "rolling3_error": None if pd.isna(r) else float(r),
            }
            for k, o, s, e, r in zip(obs.index, obs, sim, err, rolling)
        },
        "meta": {
            "data_path": data_path,
            "calibration": calib_path,
            "window": list(window),
            "n_agents": n_agents,
            "seed": seed,
        },
    }

    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))

    if verbose:
        print(f"\n  {'month':<10}{'observed':>14}{'simulated':>14}{'error':>9}"
              f"{'roll3':>9}  flag")
        for k, row in report["monthly"].items():
            r3 = row["rolling3_error"]
            flags = []
            if abs(row["pct_error"]) > thresholds.month_warn:
                flags.append("MONTH")
            if r3 is not None and abs(r3) > thresholds.drift_warn:
                flags.append("DRIFT")
            print(f"  {k:<10}{row['observed']:>14,.0f}{row['simulated']:>14,.0f}"
                  f"{row['pct_error']:>8.1%}"
                  f"{('    --' if r3 is None else f'{r3:>8.1%}'):>9}  "
                  f"{','.join(flags)}")
        print(f"\n  MAPE {mape:.2%} (fail > {thresholds.mape_fail:.0%})  "
              f"status: {status}")
        if report_path:
            print(f"  report written to {report_path}")
    return report


def exit_code(report: dict, strict: bool = False) -> int:
    if report["status"] == "FAIL":
        return 1
    if report["status"] == "WARN" and strict:
        return 1
    return 0
