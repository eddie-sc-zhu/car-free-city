"""Naive per-agent reference engine (the "v1" implementation).

This is the straightforward way to write an ABM -- a Python loop over agent
objects every day, plus a mesa-DataCollector-style history that snapshots
per-agent state each step. It is kept deliberately: it is the ground truth
the vectorized engine is verified against (same seed -> same random stream
-> statistically identical output), and it is the "before" side of the
benchmark in `carfree/benchmark.py`.

Two properties make it slow and heavy, on purpose, because they are exactly
what the optimization pass removed:

  1. the per-agent daily update runs in interpreted Python
     (`math.exp`, attribute lookups, float boxing), and
  2. `collect_agents=True` (the default here) appends a *copy* of every
     agent's state every day -- the redundant state copies that dominated
     peak memory in the original implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .engine_common import BaseEngine


class NaiveEngine(BaseEngine):
    name = "naive"

    def __init__(self, *args, collect_agents: Optional[bool] = None, **kwargs):
        self.collect_agents = True if collect_agents is None else collect_agents
        super().__init__(*args, **kwargs)

    def habit_snapshot(self) -> np.ndarray:
        return np.asarray(self.habit)

    def post_init(self) -> None:
        # per-agent state and attributes as plain Python objects
        pop = self.pop
        self.habit = [0.0] * self.n
        self._walk = pop.walk_time.tolist()
        self._ivt = pop.bus_ivt.tolist()
        self._car_time = pop.car_time.tolist()
        self._other_time = pop.other_time.tolist()
        self._vot = pop.vot.tolist()
        self._has_car = pop.has_car.tolist()
        self._downtown = pop.works_downtown.tolist()
        self.agent_history: list[dict] = []

    def step_agents(self, day, p_travel, wait, ivt_mult, policy_active,
                    u_travel, u_mode):
        p = self.params
        theta = p.theta
        asc_bus, asc_other = p.asc_bus, p.asc_other
        habit_w, habit_d = p.habit_weight, p.habit_decay
        fare = p.rider_day_fare
        op_cost = p.car_op_cost_per_min
        has_pass = self.has_pass

        riders = paying = travelers = car_users = 0
        modes = [0] * self.n   # 0 = no trip, 1 = bus, 2 = car, 3 = other

        for i in range(self.n):
            rode = False
            if u_travel[i] < p_travel:
                travelers += 1
                vot = self._vot[i]

                fare_day = 0.0 if has_pass[i] else fare
                t_bus = self._walk[i] + wait + self._ivt[i] * ivt_mult
                u_bus = (asc_bus + habit_w * self.habit[i]
                         - theta * fare_day - theta * vot * 2.0 * t_bus)

                car_ok = self._has_car[i] and not (policy_active and self._downtown[i])
                if car_ok:
                    park = (p.car_park_downtown if self._downtown[i]
                            else p.car_park_other)
                    u_car = -theta * (park + (op_cost + vot) * 2.0 * self._car_time[i])
                else:
                    u_car = -math.inf

                u_other = asc_other - theta * vot * 2.0 * self._other_time[i]

                m = max(u_bus, u_car, u_other)
                e_bus = math.exp(u_bus - m)
                e_car = math.exp(u_car - m) if car_ok else 0.0
                e_other = math.exp(u_other - m)
                denom = e_bus + e_car + e_other

                draw = u_mode[i]
                if draw < e_bus / denom:
                    rode = True
                    riders += 1
                    modes[i] = 1
                    if not has_pass[i]:
                        paying += 1
                elif draw < (e_bus + e_car) / denom:
                    car_users += 1
                    modes[i] = 2
                else:
                    modes[i] = 3

            self.habit[i] += habit_d * ((1.0 if rode else 0.0) - self.habit[i])

        if self.collect_agents:
            # mesa-DataCollector-style per-agent snapshot: a fresh copy of the
            # full agent state for every simulated day
            self.agent_history.append({
                "day": day,
                "habit": list(self.habit),
                "mode": list(modes),
                "has_pass": has_pass.tolist(),
            })

        return riders, paying, travelers, car_users
