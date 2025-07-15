"""Vectorized simulation engine (the production fast path).

The per-agent daily update -- trip decision, three-mode utility evaluation,
logit choice, habit update -- runs as ~20 NumPy array operations over the
whole population instead of a Python loop, and the engine keeps only
per-day aggregate records (no per-agent history), which together are what
cut runtime and peak memory versus the naive reference implementation in
`engine_naive.py`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .engine_common import BaseEngine
from .params import ModelParams, Scenario, scenario_by_name
from .results import SimulationResult


class VectorizedEngine(BaseEngine):
    name = "vectorized"

    def habit_snapshot(self) -> np.ndarray:
        return self.habit

    def post_init(self) -> None:
        self.habit = np.zeros(self.n)
        # hoisted invariants: everything that does not change day to day
        p, pop = self.params, self.pop
        self._u_other = p.asc_other - p.theta * (pop.vot * 2.0 * pop.other_time)
        park = np.where(pop.works_downtown, p.car_park_downtown, p.car_park_other)
        self._u_car_base = -p.theta * (
            park + (p.car_op_cost_per_min + pop.vot) * 2.0 * pop.car_time)
        self._vot2 = p.theta * pop.vot * 2.0

    def step_agents(self, day, p_travel, wait, ivt_mult, policy_active,
                    u_travel, u_mode):
        p, pop = self.params, self.pop

        traveling = u_travel < p_travel

        fare_day = np.where(self.has_pass, 0.0, p.rider_day_fare)
        t_bus = pop.walk_time + wait + pop.bus_ivt * ivt_mult
        u_bus = (p.asc_bus + p.habit_weight * self.habit
                 - p.theta * fare_day - self._vot2 * t_bus)

        car_ok = pop.has_car
        if policy_active:
            car_ok = car_ok & ~pop.works_downtown
        u_car = np.where(car_ok, self._u_car_base, -np.inf)

        m = np.maximum(np.maximum(u_bus, u_car), self._u_other)
        e_bus = np.exp(u_bus - m)
        e_car = np.exp(u_car - m)
        e_other = np.exp(self._u_other - m)
        denom = e_bus + e_car + e_other

        p_bus = e_bus / denom
        rode_bus = traveling & (u_mode < p_bus)
        used_car = traveling & ~rode_bus & (u_mode < (e_bus + e_car) / denom)

        self.habit += p.habit_decay * (rode_bus - self.habit)

        riders = int(rode_bus.sum())
        paying = int((rode_bus & ~self.has_pass).sum())
        return riders, paying, int(traveling.sum()), int(used_car.sum())


def simulate(
    scenario: Scenario | str,
    params: Optional[ModelParams] = None,
    n_agents: int = 20_000,
    n_days: int = 365 * 5,
    start: str = "2019-01-01",
    seed: int = 7,
    burn_in_days: int = 90,
    engine: str = "vectorized",
    collect_agents: Optional[bool] = None,
) -> SimulationResult:
    """Run one scenario and return a SimulationResult.

    engine="vectorized" is the fast production path; engine="naive" runs the
    per-agent reference implementation (see engine_naive.py) used for
    benchmarking and equivalence testing.
    """
    if isinstance(scenario, str):
        scenario = scenario_by_name(scenario)
    params = params or ModelParams()

    if engine == "vectorized":
        eng = VectorizedEngine(params, scenario, n_agents, n_days, start, seed,
                               burn_in_days)
    elif engine == "naive":
        from .engine_naive import NaiveEngine
        eng = NaiveEngine(params, scenario, n_agents, n_days, start, seed,
                          burn_in_days, collect_agents=collect_agents)
    else:
        raise ValueError(f"Unknown engine '{engine}' (use 'vectorized' or 'naive')")
    return eng.run()
