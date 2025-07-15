"""Behavioral sanity checks on the model's causal structure."""

import numpy as np
import pytest

from carfree.engine import simulate
from carfree.params import ModelParams

N_AGENTS = 4_000
N_DAYS = 365 * 3
SEED = 11


@pytest.fixture(scope="module")
def suite():
    names = ["status_quo", "car_free", "car_free_passes", "passes_only"]
    return {name: simulate(name, n_agents=N_AGENTS, n_days=N_DAYS, seed=SEED)
            for name in names}


def _final_year_boardings(res):
    return res.daily["boardings"].iloc[-365:].sum()


def test_no_nans_and_bounded(suite):
    for res in suite.values():
        d = res.daily
        assert not d.isna().any().any()
        assert (d["bus_riders"] <= d["travelers"]).all()
        assert (d["convenience"].between(0, 1)).all()
        assert (d["awareness"].between(0, 1)).all()
        assert (d["frequency"] >= res.params.freq_min).all()
        assert (d["frequency"] <= res.params.freq_max).all()


def test_car_ban_lifts_ridership(suite):
    assert _final_year_boardings(suite["car_free"]) > \
        1.05 * _final_year_boardings(suite["status_quo"])


def test_passes_lift_ridership_further(suite):
    assert _final_year_boardings(suite["car_free_passes"]) > \
        _final_year_boardings(suite["car_free"])
    assert _final_year_boardings(suite["passes_only"]) > \
        _final_year_boardings(suite["status_quo"])


def test_passes_cut_revenue_yield(suite):
    """Loop B2: pass riders yield less revenue per boarding than fare payers."""
    with_p = suite["car_free_passes"].kpis()
    without = suite["car_free"].kpis()
    assert with_p["revenue_per_boarding"] < without["revenue_per_boarding"]


def test_pass_adoption_monotonic(suite):
    holders = suite["car_free_passes"].daily["pass_holders"].to_numpy()
    assert (np.diff(holders) >= 0).all()
    offers = suite["car_free_passes"].daily["offer_share"].to_numpy()
    assert (np.diff(offers) >= -1e-12).all()


def test_deeper_discount_more_adoption_less_yield():
    lo = simulate("car_free_passes",
                  params=ModelParams(bulk_discount=0.1),
                  n_agents=N_AGENTS, n_days=N_DAYS, seed=SEED)
    hi = simulate("car_free_passes",
                  params=ModelParams(bulk_discount=0.8),
                  n_agents=N_AGENTS, n_days=N_DAYS, seed=SEED)
    assert hi.daily["pass_holders"].iloc[-1] > lo.daily["pass_holders"].iloc[-1]
    assert hi.kpis()["revenue_per_boarding"] < lo.kpis()["revenue_per_boarding"]


def test_fare_revenue_accounting(suite):
    """Farebox revenue == paying rider-days * day fare * scale, exactly."""
    res = suite["status_quo"]
    p = res.params
    expected = (res.daily["paying_riders"] * p.rider_day_fare
                * p.persons_per_agent)
    assert np.allclose(res.daily["fare_revenue"], expected)


def test_service_frequency_responds_to_policy(suite):
    """Loop R1: the demand shock should eventually raise service frequency."""
    assert suite["car_free"].daily["frequency"].iloc[-1] > \
        suite["status_quo"].daily["frequency"].iloc[-1]


def test_deterministic_given_seed():
    a = simulate("car_free", n_agents=1000, n_days=200, seed=3)
    b = simulate("car_free", n_agents=1000, n_days=200, seed=3)
    assert a.daily.equals(b.daily)
