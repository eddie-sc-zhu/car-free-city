"""Simulation output container and KPI summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd

from .params import ModelParams, Scenario

DAILY_COLUMNS = [
    "boardings",        # real-world boardings/day (agents * persons_per_agent)
    "bus_riders",       # agents riding the bus today
    "paying_riders",    # bus riders without an employer pass
    "travelers",        # agents making any trip today
    "car_users",        # agents driving today
    "fare_revenue",     # $ collected at the farebox today (real-world scale)
    "frequency",        # system-average buses/hour
    "wait_time",        # average wait, minutes
    "load_factor",      # boardings / capacity
    "convenience",      # composite index in [0, 1]
    "awareness",        # awareness stock in [0, 1]
    "pass_holders",     # agents holding an employer pass
    "offer_share",      # share of employers offering passes
    "policy_active",    # downtown car ban in force (0/1)
]


@dataclass
class SimulationResult:
    daily: pd.DataFrame            # indexed by date, one row per simulated day
    params: ModelParams
    scenario: Scenario
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def monthly(self) -> pd.DataFrame:
        """Monthly aggregates including employer bulk-pass revenue."""
        d = self.daily
        period = d.index.to_period("M")
        g = d.groupby(period)
        ppa = self.params.persons_per_agent
        wholesale = self.params.monthly_pass_price * (1.0 - self.params.bulk_discount)

        m = pd.DataFrame({
            "boardings": g["boardings"].sum(),
            "fare_revenue": g["fare_revenue"].sum(),
            "pass_revenue": g["pass_holders"].last() * wholesale * ppa,
            "frequency": g["frequency"].mean(),
            "wait_time": g["wait_time"].mean(),
            "load_factor": g["load_factor"].mean(),
            "convenience": g["convenience"].mean(),
            "awareness": g["awareness"].mean(),
            "bus_share": g["bus_riders"].sum() / g["travelers"].sum(),
            "pass_rider_share": g["pass_holders"].last() / self.meta.get("n_agents", np.nan),
            "offer_share": g["offer_share"].last(),
            "days": g["boardings"].count(),
        })
        m["total_revenue"] = m["fare_revenue"] + m["pass_revenue"]
        return m

    def kpis(self, window_days: int = 365) -> Dict[str, float]:
        """Headline metrics over the final `window_days` of the run."""
        d = self.daily.iloc[-window_days:]
        m = self.monthly.iloc[-max(1, window_days // 30):]
        n = self.meta.get("n_agents", np.nan)
        return {
            "annual_boardings": float(d["boardings"].sum() * (365 / len(d))),
            "annual_fare_revenue": float(m["fare_revenue"].sum() * (12 / len(m))),
            "annual_pass_revenue": float(m["pass_revenue"].sum() * (12 / len(m))),
            "annual_total_revenue": float(m["total_revenue"].sum() * (12 / len(m))),
            "mean_bus_share": float(d["bus_riders"].sum() / d["travelers"].sum()),
            "mean_convenience": float(d["convenience"].mean()),
            "mean_wait_min": float(d["wait_time"].mean()),
            "final_frequency": float(d["frequency"].iloc[-1]),
            "pass_holder_share": float(d["pass_holders"].iloc[-1] / n),
            "employer_offer_share": float(d["offer_share"].iloc[-1]),
            "revenue_per_boarding": float(m["total_revenue"].sum() / max(m["boardings"].sum(), 1.0)),
        }

    def to_csv(self, path: str) -> None:
        self.daily.to_csv(path, index_label="date")
# results