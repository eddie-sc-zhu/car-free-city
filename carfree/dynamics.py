"""Aggregate system dynamics shared by both engines.

These pure scalar functions encode the causal loops of the model:

  R1 (reinforcing): ridership -> fare revenue -> service frequency ->
                    shorter waits -> convenience -> ridership
  R2 (reinforcing): convenience & rider share -> awareness ->
                    employer pass adoption -> ridership
  B1 (balancing):   ridership -> crowding -> longer in-vehicle time &
                    lower convenience -> ridership
  B2 (balancing):   employer passes -> discounted bulk revenue ->
                    lower farebox recovery -> service pressure
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .params import ModelParams


def wait_time(freq: float) -> float:
    """Average wait in minutes: half the headway of `freq` buses/hour."""
    return 30.0 / freq


def crowding_ivt_multiplier(params: ModelParams, load: float) -> float:
    """In-vehicle time inflation once demand exceeds capacity (loop B1)."""
    return 1.0 + params.crowd_ivt_penalty * max(0.0, load - 1.0)


def convenience_index(params: ModelParams, freq: float, load: float) -> float:
    """Composite rider-convenience score in [0, 1].

    Blends wait time, crowding, and network coverage (proxied by frequency).
    Used for reporting and as a driver of awareness; the mode-choice utility
    already prices wait and crowding directly, so this index does not feed
    back into utilities (no double counting).
    """
    wait_component = math.exp(-wait_time(freq) / 12.0)
    crowd_component = 1.0 - min(max((load - 0.80) / 0.70, 0.0), 1.0)
    coverage_component = min(freq / 10.0, 1.0)
    return 0.45 * wait_component + 0.35 * crowd_component + 0.20 * coverage_component


def awareness_step(params: ModelParams, awareness: float, convenience: float,
                   rider_share: float) -> float:
    """Daily relaxation of public awareness toward its drivers (loop R2)."""
    target = 0.5 * convenience + 0.5 * min(1.0, rider_share / params.rider_share_norm)
    a = awareness + params.awareness_rate * (target - awareness)
    return min(max(a, 0.0), 1.0)


def service_adjustment(params: ModelParams, freq: float, recovery: float,
                       mean_load: float) -> float:
    """Monthly operator response (loops R1 and B2).

    Frequency grows when farebox recovery beats target and buses run full;
    it is cut when recovery sags, bounded by max_service_change per month.
    """
    gap_recovery = (recovery - params.target_recovery) / params.target_recovery
    gap_load = mean_load - params.target_load
    growth = params.k_recovery * gap_recovery + params.k_load * gap_load
    growth = min(max(growth, -params.max_service_change), params.max_service_change)
    return min(max(freq * (1.0 + growth), params.freq_min), params.freq_max)


@dataclass
class ServiceEconomics:
    """Capacity and cost coefficients fixed at the end of burn-in.

    Anchoring both to the model's own burn-in equilibrium makes the service
    feedback neutral at baseline: absent a policy shock, recovery == target
    and load == target, so frequency holds steady.
    """
    capacity_per_freq: float      # boardings/day (agent units) per bus/hour
    unit_cost_per_freq: float     # $/month (agent units) per bus/hour

    @classmethod
    def from_burn_in(cls, params: ModelParams, freq: float,
                     mean_daily_boardings: float,
                     mean_daily_revenue: float) -> "ServiceEconomics":
        capacity = mean_daily_boardings / (freq * params.target_load)
        monthly_revenue = mean_daily_revenue * 30.44
        unit_cost = monthly_revenue / (freq * params.target_recovery)
        return cls(capacity_per_freq=capacity, unit_cost_per_freq=unit_cost)

    def load_factor(self, freq: float, boardings: float) -> float:
        return boardings / (freq * self.capacity_per_freq)

    def recovery(self, freq: float, monthly_revenue: float, days: int) -> float:
        monthly_cost = self.unit_cost_per_freq * freq * (days / 30.44)
        return monthly_revenue / monthly_cost if monthly_cost > 0 else 0.0
