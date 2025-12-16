"""Synthetic commuter population.

One agent represents `params.persons_per_agent` real Baltimore travelers.
Attributes are drawn once at initialization in a *fixed order* so that the
naive and vectorized engines, given the same seed, operate on identical
populations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import ModelParams


@dataclass
class Population:
    """Struct-of-arrays agent container (the vectorized layout)."""
    n: int
    employer_id: np.ndarray      # int32, -1 for agents without an employer
    has_car: np.ndarray          # bool
    works_downtown: np.ndarray   # bool
    vot: np.ndarray              # float64, $/minute
    walk_time: np.ndarray        # float64, minutes one-way (access + egress)
    bus_ivt: np.ndarray          # float64, minutes one-way in-vehicle
    car_time: np.ndarray         # float64, minutes one-way
    other_time: np.ndarray       # float64, minutes one-way (walk / bike)
    pass_propensity: np.ndarray  # float64 in [0, 1)
    n_employers: int


def build_population(params: ModelParams, n: int, rng: np.random.Generator) -> Population:
    """Draw the synthetic population. Draw order is part of the contract."""
    employed = rng.random(n) < params.employed_share

    n_employers = max(1, int(round(n * params.employed_share / params.mean_employer_size)))
    sizes = rng.lognormal(mean=0.0, sigma=1.0, size=n_employers)
    weights = sizes / sizes.sum()
    employer_id = rng.choice(n_employers, size=n, p=weights).astype(np.int32)
    employer_id[~employed] = -1

    has_car = rng.random(n) < params.car_access_share
    works_downtown = rng.random(n) < params.works_downtown_share

    mu = math.log(params.vot_mean) - 0.5 * params.vot_sigma**2
    vot = rng.lognormal(mu, params.vot_sigma, n)

    walk_time = rng.uniform(*params.walk_time_range, n)
    ivt_mu = math.log(params.bus_ivt_mean) - 0.5 * params.bus_ivt_sigma**2
    bus_ivt = rng.lognormal(ivt_mu, params.bus_ivt_sigma, n)
    car_time = bus_ivt * rng.uniform(*params.car_time_ratio, n)
    other_time = bus_ivt * rng.uniform(*params.other_time_ratio, n)
    pass_propensity = rng.random(n)

# pop
    return Population(
        n=n,
        employer_id=employer_id,
        has_car=has_car,
        works_downtown=works_downtown,
        vot=vot,
        walk_time=walk_time,
        bus_ivt=bus_ivt,
        car_time=car_time,
        other_time=other_time,
        pass_propensity=pass_propensity,
        n_employers=n_employers,
    )
