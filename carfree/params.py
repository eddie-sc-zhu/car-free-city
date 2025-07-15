"""Model parameters and scenario definitions.

All dollar figures are real-world anchored:
  * $2.00 one-way local bus fare, $4.40 day pass, $74 monthly pass
    (MDOT MTA fare tariff for Baltimore core services).
  * ~30% of Baltimore City households have no vehicle available
    (ACS via Baltimore Neighborhood Indicators Alliance); commuters skew
    toward car access, hence car_access_share > 0.62 at the person level.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelParams:
    # ---- demand / behavior -------------------------------------------------
    base_travel_prob: float = 0.84         # weekday probability an agent makes a trip
    dow_factors: Tuple[float, ...] = (1.02, 1.05, 1.05, 1.03, 1.00, 0.62, 0.48)
    seasonal_factors: Tuple[float, ...] = (1.0,) * 12   # Jan..Dec, calibrated
    asc_bus: float = -1.20                 # alternative-specific constant, calibrated
    asc_other: float = -0.45               # walk / bike / stay-local alternative
    theta: float = 0.09                    # utility per generalized dollar
    habit_weight: float = 0.50             # utility bonus at habit == 1
    habit_decay: float = 0.05              # daily exponential habit update rate
    vot_mean: float = 0.25                 # value of time, $/minute (~$15/hr)
    vot_sigma: float = 0.45                # lognormal sigma of value of time

    # ---- synthetic population ----------------------------------------------
    employed_share: float = 0.72           # agents attached to an employer
    car_access_share: float = 0.62         # person-level car availability
    works_downtown_share: float = 0.45     # jobs inside the car-free zone
    walk_time_range: Tuple[float, float] = (2.0, 12.0)     # access+egress, min one-way
    bus_ivt_mean: float = 28.0             # in-vehicle time, min one-way
    bus_ivt_sigma: float = 0.35
    car_time_ratio: Tuple[float, float] = (0.45, 0.75)     # car time / bus time
    other_time_ratio: Tuple[float, float] = (1.6, 3.2)     # walk-bike time / bus time

    # ---- fares & money -----------------------------------------------------
    rider_day_fare: float = 4.40           # effective $ per paying rider-day (day pass)
    monthly_pass_price: float = 74.0       # MTA monthly pass, $
    bulk_discount: float = 0.35            # employer buys passes at (1 - discount)
    transfer_rate: float = 0.25            # extra boardings per trip from transfers
    car_park_downtown: float = 11.0        # $/day parking in the zone
    car_park_other: float = 3.5            # $/day parking elsewhere
    car_op_cost_per_min: float = 0.20      # fuel + wear, $/minute driven

    # ---- service supply (operator) ----------------------------------------
    freq0: float = 4.0                     # system-average buses/hour at start
    freq_min: float = 1.5
    freq_max: float = 14.0
    target_load: float = 0.75              # boardings / capacity the operator aims for
    target_recovery: float = 0.25          # farebox recovery ratio the operator aims for
    k_recovery: float = 0.15               # service response to recovery gap
    k_load: float = 0.10                   # service response to crowding gap
    max_service_change: float = 0.05       # max +/- frequency change per month
    crowd_ivt_penalty: float = 0.25        # in-vehicle time inflation when load > 1

    # ---- employer transit-pass program -------------------------------------
    mean_employer_size: int = 40
    initial_offer_share: float = 0.04      # employers offering passes pre-program
    adopt_hazard0: float = 0.008           # monthly base adoption hazard
    adopt_awareness_gain: float = 2.0      # hazard multiplier slope on awareness
    adopt_peer_gain: float = 1.5           # hazard multiplier slope on peer share
    adopt_hazard_cap: float = 0.10         # monthly hazard ceiling
    adopt_discount_base: float = 0.4       # hazard multiplier at zero bulk discount
    adopt_discount_gain: float = 1.7       # hazard multiplier slope on bulk discount
    enroll_base: float = 0.75              # max employee enrollment propensity cutoff

    # ---- awareness ----------------------------------------------------------
    awareness0: float = 0.35
    awareness_rate: float = 0.02           # daily relaxation rate toward its driver
    rider_share_norm: float = 0.35         # rider share that saturates awareness

    # ---- scale --------------------------------------------------------------
    persons_per_agent: float = 27.0        # real travelers per agent, calibrated

    def with_updates(self, **kwargs) -> "ModelParams":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Scenario:
    """A policy scenario layered on top of ModelParams."""
    name: str
    description: str
    policy_start_day: Optional[int] = None   # day index when the downtown car ban starts
    employer_program: bool = False           # active employer pass-adoption process

    def policy_active(self, day: int) -> bool:
        return self.policy_start_day is not None and day >= self.policy_start_day


SCENARIOS = {
    "status_quo": Scenario(
        name="status_quo",
        description="No car-free policy, no employer pass program (calibration baseline).",
    ),
    "car_free": Scenario(
        name="car_free",
        description="Downtown car ban starting at day 365; no employer pass program.",
        policy_start_day=365,
    ),
    "car_free_passes": Scenario(
        name="car_free_passes",
        description="Downtown car ban at day 365 plus stochastic employer pass adoption.",
        policy_start_day=365,
        employer_program=True,
    ),
    "passes_only": Scenario(
        name="passes_only",
        description="Employer pass adoption without the car ban (isolates the pass lift).",
        employer_program=True,
    ),
}


def scenario_by_name(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        options = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario '{name}'. Options: {options}") from None
