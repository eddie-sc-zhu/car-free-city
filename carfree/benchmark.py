"""Naive vs vectorized engine benchmark (runtime + peak memory).

Runtime is measured on a clean pass; peak memory on a second pass under
tracemalloc (tracemalloc adds overhead, so the two are never mixed). Both
engines consume identical random streams, so they do identical *model* work
-- the measured gap is purely implementation: the interpreted per-agent
loop and the mesa-style per-day agent-state copies on the naive side.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import List, Optional, Sequence

from .engine import simulate
from .params import ModelParams


def _measure(engine: str, n_agents: int, n_days: int, seed: int,
             params: ModelParams) -> dict:
    kwargs = dict(params=params, n_agents=n_agents, n_days=n_days,
                  seed=seed, engine=engine, burn_in_days=30)

    t0 = time.perf_counter()
    result = simulate("car_free_passes", **kwargs)
    runtime = time.perf_counter() - t0

    tracemalloc.start()
    simulate("car_free_passes", **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "engine": engine,
        "n_agents": n_agents,
        "n_days": n_days,
        "runtime_s": runtime,
        "peak_mem_mb": peak / 1e6,
        "boardings_total": float(result.daily["boardings"].sum()),
    }


def run_benchmark(sizes: Sequence[int] = (1_000, 2_000, 4_000),
                  n_days: int = 180, seed: int = 7,
                  params: Optional[ModelParams] = None,
                  verbose: bool = True) -> List[dict]:
    params = params or ModelParams()
    rows: List[dict] = []
    for n in sizes:
        naive = _measure("naive", n, n_days, seed, params)
        fast = _measure("vectorized", n, n_days, seed, params)
        naive["speedup"] = fast["speedup"] = naive["runtime_s"] / fast["runtime_s"]
        naive["mem_ratio"] = fast["mem_ratio"] = (naive["peak_mem_mb"]
                                                  / fast["peak_mem_mb"])
        rows += [naive, fast]
        if verbose:
            print(f"  n={n:>7,} days={n_days}: "
                  f"naive {naive['runtime_s']:7.2f}s / {naive['peak_mem_mb']:8.1f} MB | "
                  f"vectorized {fast['runtime_s']:6.3f}s / {fast['peak_mem_mb']:6.1f} MB "
                  f"-> {naive['speedup']:.1f}x faster, "
                  f"{naive['mem_ratio']:.0f}x less memory")
    return rows


def city_scale_demo(n_agents: int = 150_000, years: int = 5, seed: int = 7,
                    params: Optional[ModelParams] = None,
                    verbose: bool = True) -> dict:
    """Show the vectorized engine handles multi-year, city-scale runs."""
    params = params or ModelParams()
    t0 = time.perf_counter()
    result = simulate("car_free_passes", params=params, n_agents=n_agents,
                      n_days=365 * years, seed=seed)
    runtime = time.perf_counter() - t0
    row = {"engine": "vectorized", "n_agents": n_agents, "n_days": 365 * years,
           "runtime_s": runtime,
           "boardings_total": float(result.daily["boardings"].sum())}
    if verbose:
        print(f"  city scale: n={n_agents:,} agents x {years} years -> "
              f"{runtime:.1f}s (vectorized)")
    return row
