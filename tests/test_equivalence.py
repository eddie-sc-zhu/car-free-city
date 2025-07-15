"""The naive per-agent engine and the vectorized engine must agree.

Both engines consume identical random streams by construction, so with the
same seed their outputs differ only by floating-point library rounding
(math.exp vs np.exp), which can flip a mode choice only when a draw lands
within ~1 ulp of a choice boundary. Aggregates must therefore match to a
tight tolerance.
"""

import numpy as np
import pytest

from carfree.engine import simulate

N_AGENTS = 800
N_DAYS = 150
SEED = 42


@pytest.fixture(scope="module")
def pair():
    fast = simulate("car_free_passes", n_agents=N_AGENTS, n_days=N_DAYS,
                    seed=SEED, engine="vectorized", burn_in_days=30)
    naive = simulate("car_free_passes", n_agents=N_AGENTS, n_days=N_DAYS,
                     seed=SEED, engine="naive", burn_in_days=30)
    return fast, naive


def test_total_boardings_match(pair):
    fast, naive = pair
    a, b = fast.daily["boardings"].sum(), naive.daily["boardings"].sum()
    assert a == pytest.approx(b, rel=0.01)


def test_daily_series_track_each_other(pair):
    fast, naive = pair
    f = fast.daily["bus_riders"].to_numpy()
    n = naive.daily["bus_riders"].to_numpy()
    assert np.corrcoef(f, n)[0, 1] > 0.99
    assert np.abs(f - n).mean() < 0.02 * max(f.mean(), 1)


def test_pass_adoption_matches(pair):
    fast, naive = pair
    assert abs(fast.daily["pass_holders"].iloc[-1]
               - naive.daily["pass_holders"].iloc[-1]) <= N_AGENTS * 0.01
    assert fast.daily["offer_share"].iloc[-1] == pytest.approx(
        naive.daily["offer_share"].iloc[-1], abs=0.02)


def test_revenue_matches(pair):
    fast, naive = pair
    assert fast.daily["fare_revenue"].sum() == pytest.approx(
        naive.daily["fare_revenue"].sum(), rel=0.01)
