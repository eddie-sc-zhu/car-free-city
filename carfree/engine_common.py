"""Shared simulation scaffold.

Both engines inherit calendar handling, burn-in anchoring, operator economics,
employer updates, and metric recording from BaseEngine; they differ ONLY in
`step_agents`, the per-agent daily update (the hot path). The random draw
order is a hard contract shared by both engines:

    1. build_population(seed)
    2. initial employer offers
    3. initial enrollment sweep (one E-sized uniform draw)
    4. each day: one n-sized draw for trip-making, one n-sized draw for mode
    5. each month end: one E-sized draw for employer adoption

Given the same seed, both engines therefore consume identical random
streams and produce statistically identical trajectories (bitwise up to
libm rounding), which is what makes the benchmark apples-to-apples.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from . import dynamics
from .employers import initial_offers, monthly_employer_update
from .params import ModelParams, Scenario
from .population import build_population
from .results import DAILY_COLUMNS, SimulationResult

ANCHOR_WINDOW_DAYS = 28   # trailing window used to anchor service economics


class BaseEngine:
    name = "base"

    def __init__(self, params: ModelParams, scenario: Scenario, n_agents: int,
                 n_days: int, start: str, seed: int, burn_in_days: int = 90):
        self.params = params
        self.scenario = scenario
        self.n = n_agents
        self.burn_in = burn_in_days
        self.total_days = burn_in_days + n_days
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.pop = build_population(params, n_agents, self.rng)
        self.offers = initial_offers(params, self.pop.n_employers, self.rng)
        self.has_pass = np.zeros(n_agents, dtype=bool)

        # calendar: burn-in precedes the reporting window
        start_ts = pd.Timestamp(start) - pd.Timedelta(days=burn_in_days)
        self.dates = pd.date_range(start_ts, periods=self.total_days, freq="D")
        self.month_of_day = self.dates.month.to_numpy()
        self.dow_of_day = self.dates.weekday.to_numpy()
        self.is_month_end = self.dates.is_month_end

        # aggregate state
        self.freq = params.freq0
        self.awareness = params.awareness0
        self.load_prev = params.target_load
        self.econ: Optional[dynamics.ServiceEconomics] = None

        # engine-agnostic per-day records (real-world units; the ONLY history kept
        # by the vectorized engine)
        self.rec = {c: np.zeros(self.total_days) for c in DAILY_COLUMNS}

        self.post_init()

        # initial enrollment sweep against ambient offers
        monthly_employer_update(
            params, self.offers, self.has_pass, self.pop.employer_id,
            self.pop.pass_propensity, self.habit_snapshot(), self.awareness,
            self.rng, program_active=False,
        )

    # ---- hooks implemented by subclasses ------------------------------------
    def post_init(self) -> None:
        """Set up engine-specific agent state (habit etc.)."""

    def habit_snapshot(self) -> np.ndarray:
        """Current per-agent habit as an ndarray (for the employer update)."""
        raise NotImplementedError

    def step_agents(self, day: int, p_travel: float, wait: float, ivt_mult: float,
                    policy_active: bool, u_travel: np.ndarray, u_mode: np.ndarray):
        """Run one day of agent decisions.

        Returns (bus_riders, paying_riders, travelers, car_users) as ints.
        Must also update habit and any engine-local state.
        """
        raise NotImplementedError

    # ---- main loop -----------------------------------------------------------
    def run(self) -> SimulationResult:
        p = self.params
        t0 = time.perf_counter()
        month_fare_agents = 0.0   # farebox revenue this month, agent units
        month_load_sum = 0.0
        month_days = 0

        for d in range(self.total_days):
            report_day = d - self.burn_in
            policy = report_day >= 0 and self.scenario.policy_active(report_day)

            wait = dynamics.wait_time(self.freq)
            ivt_mult = dynamics.crowding_ivt_multiplier(p, self.load_prev)
            conv = dynamics.convenience_index(p, self.freq, self.load_prev)

            season = p.seasonal_factors[self.month_of_day[d] - 1]
            dowf = p.dow_factors[self.dow_of_day[d]]
            p_travel = min(p.base_travel_prob * season * dowf, 0.98)

            u_travel = self.rng.random(self.n)
            u_mode = self.rng.random(self.n)
            riders, paying, travelers, car_users = self.step_agents(
                d, p_travel, wait, ivt_mult, policy, u_travel, u_mode)

            boardings_agents = riders * 2.0 * (1.0 + p.transfer_rate)
            fare_agents = paying * p.rider_day_fare
            if self.econ is not None:
                load = self.econ.load_factor(self.freq, boardings_agents)
            else:
                load = p.target_load
            rider_share = riders / max(travelers, 1)
            self.awareness = dynamics.awareness_step(p, self.awareness, conv, rider_share)

            r = self.rec
            r["boardings"][d] = boardings_agents * p.persons_per_agent
            r["bus_riders"][d] = riders
            r["paying_riders"][d] = paying
            r["travelers"][d] = travelers
            r["car_users"][d] = car_users
            r["fare_revenue"][d] = fare_agents * p.persons_per_agent
            r["frequency"][d] = self.freq
            r["wait_time"][d] = wait
            r["load_factor"][d] = load
            r["convenience"][d] = conv
            r["awareness"][d] = self.awareness
            r["pass_holders"][d] = self.has_pass.sum()
            r["offer_share"][d] = self.offers.mean()
            r["policy_active"][d] = float(policy)

            month_fare_agents += fare_agents
            month_load_sum += load
            month_days += 1
            self.load_prev = load

            # anchor operator economics to the model's own burn-in equilibrium
            if d == self.burn_in - 1:
                w = min(ANCHOR_WINDOW_DAYS, d + 1)
                mean_boardings = float(
                    r["boardings"][d - w + 1: d + 1].mean()) / p.persons_per_agent
                wholesale = p.monthly_pass_price * (1.0 - p.bulk_discount)
                pass_daily = self.has_pass.sum() * wholesale / 30.44
                mean_revenue = float(
                    r["fare_revenue"][d - w + 1: d + 1].mean()) / p.persons_per_agent
                self.econ = dynamics.ServiceEconomics.from_burn_in(
                    p, self.freq, mean_boardings, mean_revenue + pass_daily)

            if self.is_month_end[d]:
                wholesale = p.monthly_pass_price * (1.0 - p.bulk_discount)
                month_revenue = month_fare_agents + self.has_pass.sum() * wholesale
                if self.econ is not None:
                    recovery = self.econ.recovery(self.freq, month_revenue, month_days)
                    mean_load = month_load_sum / month_days
                    self.freq = dynamics.service_adjustment(p, self.freq, recovery, mean_load)
                monthly_employer_update(
                    p, self.offers, self.has_pass, self.pop.employer_id,
                    self.pop.pass_propensity, self.habit_snapshot(),
                    self.awareness, self.rng,
                    program_active=self.scenario.employer_program and report_day >= 0,
                )
                month_fare_agents = 0.0
                month_load_sum = 0.0
                month_days = 0

        runtime = time.perf_counter() - t0
        daily = pd.DataFrame(
            {c: self.rec[c][self.burn_in:] for c in DAILY_COLUMNS},
            index=self.dates[self.burn_in:],
        )
        return SimulationResult(
            daily=daily, params=p, scenario=self.scenario,
            meta={
                "engine": self.name, "n_agents": self.n, "seed": self.seed,
                "burn_in_days": self.burn_in, "runtime_s": runtime,
            },
        )
