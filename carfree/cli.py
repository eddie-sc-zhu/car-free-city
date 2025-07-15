"""Command-line interface: python -m carfree <command>."""

from __future__ import annotations

import argparse
import sys


def _add_common(p, agents=20_000, years=5, seed=7):
    p.add_argument("--agents", type=int, default=agents)
    p.add_argument("--years", type=int, default=years)
    p.add_argument("--seed", type=int, default=seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="carfree",
        description="Agent-based model of Baltimore bus ridership under a "
                    "car-free-city policy.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run one scenario and print KPIs")
    p.add_argument("--scenario", default="car_free_passes")
    p.add_argument("--engine", default="vectorized",
                   choices=["vectorized", "naive"])
    p.add_argument("--csv", default=None, help="write daily output to CSV")
    _add_common(p)

    p = sub.add_parser("calibrate", help="fit the model to observed MTA data")
    p.add_argument("--data", default=None)
    p.add_argument("--agents", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("validate",
                       help="flag divergence between simulated and observed "
                            "ridership (CI-friendly exit code)")
    p.add_argument("--data", default=None)
    p.add_argument("--strict", action="store_true",
                   help="treat WARN as failure")

    p = sub.add_parser("dashboard",
                       help="run the scenario suite + discount sweep and "
                            "render the stakeholder dashboard")
    p.add_argument("--out", default="outputs/dashboard.png")
    _add_common(p)

    p = sub.add_parser("sweep", help="bulk-discount tradeoff sweep (CSV + table)")
    p.add_argument("--values", default="0,0.2,0.35,0.5,0.65,0.8")
    p.add_argument("--csv", default="outputs/discount_sweep.csv")
    _add_common(p)

    p = sub.add_parser("benchmark", help="naive vs vectorized engine benchmark")
    p.add_argument("--sizes", default="1000,2000,4000")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--city-scale", action="store_true",
                   help="also time a 150k-agent, 5-year vectorized run")
    p.add_argument("--plot", default=None,
                   help="write a benchmark chart PNG to this path")

    args = parser.parse_args(argv)

    if args.command == "run":
        from .calibration import calibrated_params
        from .engine import simulate
        params = calibrated_params()
        res = simulate(args.scenario, params=params, n_agents=args.agents,
                       n_days=365 * args.years, seed=args.seed,
                       engine=args.engine)
        print(f"scenario: {res.scenario.name} - {res.scenario.description}")
        print(f"engine:   {res.meta['engine']}  agents: {args.agents:,}  "
              f"runtime: {res.meta['runtime_s']:.2f}s")
        print("\nfinal-year KPIs:")
        for k, v in res.kpis().items():
            print(f"  {k:<26} {v:>16,.4g}")
        if args.csv:
            res.to_csv(args.csv)
            print(f"\ndaily output written to {args.csv}")
        return 0

    if args.command == "calibrate":
        from .calibration import DEFAULT_DATA, calibrate
        calibrate(data_path=args.data or DEFAULT_DATA, n_agents=args.agents,
                  seed=args.seed)
        return 0

    if args.command == "validate":
        from .calibration import DEFAULT_DATA
        from .validation import exit_code, run_validation
        report = run_validation(data_path=args.data or DEFAULT_DATA)
        return exit_code(report, strict=args.strict)

    if args.command == "dashboard":
        from pathlib import Path

        from .calibration import calibrated_params
        from .dashboard import (make_dashboard, run_suite, sweep_bulk_discount)
        params = calibrated_params()
        print("running scenario suite...")
        results = run_suite(params, args.agents, args.years, args.seed)
        print("running bulk-discount sweep...")
        sweep = sweep_bulk_discount(params, n_agents=args.agents,
                                    years=args.years, seed=args.seed,
                                    reference=results["car_free"])
        validation = None
        report = Path("outputs/validation_report.json")
        if report.exists():
            import json
            validation = json.loads(report.read_text())
        png = make_dashboard(results, sweep, out_png=args.out,
                             validation=validation)
        print(f"dashboard written to {png} (+ .html report)")
        return 0

    if args.command == "sweep":
        from .calibration import calibrated_params
        from .dashboard import sweep_bulk_discount
        params = calibrated_params()
        values = [float(v) for v in args.values.split(",")]
        df = sweep_bulk_discount(params, values, n_agents=args.agents,
                                 years=args.years, seed=args.seed)
        if args.csv:
            from pathlib import Path
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.csv, index=False)
            print(f"sweep written to {args.csv}")
        return 0

    if args.command == "benchmark":
        from .benchmark import city_scale_demo, run_benchmark
        sizes = [int(s) for s in args.sizes.split(",")]
        rows = run_benchmark(sizes, n_days=args.days, seed=args.seed)
        if args.city_scale:
            city_scale_demo(seed=args.seed)
        if args.plot:
            from .dashboard import make_benchmark_chart
            print(f"benchmark chart written to {make_benchmark_chart(rows, args.plot)}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
