"""Agent-based complex-systems model of Baltimore bus ridership under a
car-free-city policy.

Public entry points:
    simulate()            -- run a scenario with either engine
    ModelParams, Scenario -- configuration dataclasses
    SCENARIOS             -- named scenario presets
"""

from .params import ModelParams, Scenario, SCENARIOS, scenario_by_name
from .engine import simulate
from .results import SimulationResult

__version__ = "2.0.0"

__all__ = [
    "ModelParams",
    "Scenario",
    "SCENARIOS",
    "scenario_by_name",
    "simulate",
    "SimulationResult",
    "__version__",
]
