"""Employer transit-pass program: a stochastic monthly adoption process.

Each month, every employer that does not yet offer subsidized passes adopts
the program with hazard

    h = h0 * (1 + g_a * awareness) * (1 + g_p * offer_share)      (capped)

so adoption follows an S-curve driven by rider awareness (word of mouth /
visible service quality) and peer imitation. Employees enroll when their
static propensity falls below an awareness-scaled cutoff; enrollment is
monotonic (nobody hands the pass back).

This module is shared verbatim by both simulation engines so their
stochastic streams stay aligned.
"""

from __future__ import annotations

import numpy as np

from .params import ModelParams


def initial_offers(params: ModelParams, n_employers: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Pre-program ambient share of employers already offering passes."""
    return rng.random(n_employers) < params.initial_offer_share


def enrollment_cutoff(params: ModelParams, awareness: float,
                      habit: np.ndarray) -> np.ndarray:
    """Per-agent enrollment cutoff.

    Enrollment skews strongly toward agents who already ride (high habit):
    commuters sign up for a transit benefit they expect to use. The habit
    term is what makes the pass program dilutive for the operator -- most
    enrollees are riders whose farebox payments get replaced by the
    discounted bulk rate -- while the nonzero floor still lets some
    occasional riders enroll and become new ridership.
    """
    return (params.enroll_base * (0.5 + 0.5 * awareness)
            * (0.25 + 0.75 * habit))


def monthly_employer_update(
    params: ModelParams,
    offers: np.ndarray,          # bool (n_employers,), mutated in place
    has_pass: np.ndarray,        # bool (n_agents,), mutated in place
    employer_id: np.ndarray,     # int32 (n_agents,), -1 = no employer
    pass_propensity: np.ndarray, # float64 (n_agents,)
    habit: np.ndarray,           # float64 (n_agents,)
    awareness: float,
    rng: np.random.Generator,
    program_active: bool,
) -> None:
    """One month of employer adoption + employee enrollment.

    The adoption uniforms are drawn unconditionally so that runs with and
    without the program share identical random streams (common random
    numbers -> low-variance scenario deltas).
    """
    u = rng.random(offers.shape[0])
    if program_active:
        offer_share = offers.mean()
        # a deeper bulk discount makes the program cheaper for employers,
        # raising the adoption hazard -- the lever behind the ridership-lift /
        # fare-revenue tradeoff curve
        discount_factor = (params.adopt_discount_base
                           + params.adopt_discount_gain * params.bulk_discount)
        hazard = (params.adopt_hazard0
                  * (1.0 + params.adopt_awareness_gain * awareness)
                  * (1.0 + params.adopt_peer_gain * offer_share)
                  * discount_factor)
        hazard = min(hazard, params.adopt_hazard_cap)
        offers |= (~offers) & (u < hazard)

    employed = employer_id >= 0
    offered = np.zeros(employer_id.shape[0], dtype=bool)
    offered[employed] = offers[employer_id[employed]]
    cutoff = enrollment_cutoff(params, awareness, habit)
    has_pass |= offered & (pass_propensity < cutoff)
