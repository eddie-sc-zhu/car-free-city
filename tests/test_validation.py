"""The divergence flagger must fire on drifted data and stay quiet on a fit."""

import json

import pandas as pd
import pytest

from carfree.calibration import (calibrate, load_observed,
                                 seasonal_factors_from_observed)
from carfree.validation import Thresholds, exit_code, run_validation

N_AGENTS = 4_000
SEED = 5


@pytest.fixture(scope="module")
def calib_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("calib") / "calibration.json"
    calibrate(n_agents=N_AGENTS, n_fit_agents=2_000, seed=SEED,
              out_path=str(path), verbose=False)
    return str(path)


def test_observed_loader():
    obs = load_observed()
    assert obs.index.freqstr in ("M", "ME")
    assert obs.loc[pd.Period("2019-10", "M")] == pytest.approx(8_170_442)


def test_seasonal_factors_mean_one():
    factors = seasonal_factors_from_observed(load_observed())
    assert sum(factors) / 12 == pytest.approx(1.0, abs=1e-6)


def test_calibrated_baseline_passes(calib_file, tmp_path):
    report = run_validation(calib_path=calib_file, n_agents=N_AGENTS,
                            report_path=str(tmp_path / "report.json"),
                            verbose=False)
    assert report["status"] in ("PASS", "WARN")
    assert report["mape"] < 0.08
    assert exit_code(report) == 0
    assert (tmp_path / "report.json").exists()


def test_flags_divergent_data(calib_file, tmp_path):
    """Shift the observed series 25% up: every check must fire."""
    obs = load_observed()
    shifted = tmp_path / "shifted.csv"
    df = pd.DataFrame({
        "Date": obs.index.strftime("%b %Y"),
        "Ridership": (obs * 1.25).round().astype(int),
    })
    df.to_csv(shifted, index=False)

    report = run_validation(calib_path=calib_file, data_path=str(shifted),
                            n_agents=N_AGENTS,
                            report_path=str(tmp_path / "report.json"),
                            verbose=False)
    assert report["status"] == "FAIL"
    assert exit_code(report) == 1
    assert len(report["flagged_months"]) == 12
    assert len(report["drift_months"]) > 0


def test_strict_mode_escalates_warn():
    report = {"status": "WARN"}
    assert exit_code(report) == 0
    assert exit_code(report, strict=True) == 1
    assert exit_code({"status": "FAIL"}) == 1
    assert exit_code({"status": "PASS"}, strict=True) == 0


def test_calibration_file_contents(calib_file):
    calib = json.loads(open(calib_file).read())
    assert calib["persons_per_agent"] > 0
    assert len(calib["seasonal_factors"]) == 12
    assert calib["fit"]["mape"] < 0.08
