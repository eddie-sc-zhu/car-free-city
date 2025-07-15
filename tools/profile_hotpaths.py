"""Profile the simulation hot paths.

Usage (from the repo root):
    python tools/profile_hotpaths.py [--agents 2000] [--days 120]

Runs cProfile over both engines and prints the top functions by cumulative
time; if line_profiler is installed (pip install line_profiler), also
line-profiles the naive engine's per-agent update -- the loop the
vectorized engine replaces. Raw .pstats files are written to outputs/ for
snakeviz or pstats.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carfree.engine import simulate  # noqa: E402


def profile_engine(engine: str, n_agents: int, n_days: int, seed: int) -> None:
    out = Path("outputs") / f"profile_{engine}.pstats"
    out.parent.mkdir(exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    simulate("car_free_passes", n_agents=n_agents, n_days=n_days, seed=seed,
             engine=engine, burn_in_days=30)
    prof.disable()
    prof.dump_stats(out)

    print(f"\n{'=' * 74}\ncProfile: {engine} engine "
          f"({n_agents:,} agents x {n_days} days) -> {out}\n{'=' * 74}")
    stats = pstats.Stats(prof)
    stats.sort_stats("cumulative").print_stats(12)


def line_profile_naive(n_agents: int, n_days: int, seed: int) -> None:
    try:
        from line_profiler import LineProfiler
    except ImportError:
        print("\n(line_profiler not installed -- `pip install line_profiler` "
              "for per-line stats on the naive agent loop)")
        return

    from carfree.engine_naive import NaiveEngine

    lp = LineProfiler()
    lp.add_function(NaiveEngine.step_agents)
    wrapped = lp(simulate)
    wrapped("car_free_passes", n_agents=n_agents, n_days=n_days, seed=seed,
            engine="naive", burn_in_days=30)
    print(f"\n{'=' * 74}\nline_profiler: NaiveEngine.step_agents\n{'=' * 74}")
    lp.print_stats()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=2000)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    profile_engine("naive", args.agents, args.days, args.seed)
    profile_engine("vectorized", args.agents, args.days, args.seed)
    line_profile_naive(args.agents, args.days, args.seed)
