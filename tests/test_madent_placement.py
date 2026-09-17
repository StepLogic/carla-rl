"""madent: MAD spikes with endpoints off, through node_placement.place."""
import numpy as np

from vae_nav.node_placement import mad_spikes, place


def test_mad_spikes_finds_the_spike_despite_the_start_transient():
    """The transient does not INFLATE the baseline (median/MAD are robust), so
    the real spike is still detected at z >= 3. The transient itself is a
    genuine excursion and may also fire; endpoints are never ADDED."""
    n = 400
    rng = np.random.default_rng(0)
    h = 0.8 + rng.uniform(-0.05, 0.05, n)    # MAD > 0, bounded jitter
    h[:20] = np.linspace(1.8, 0.8, 20)      # episode-start transient
    h[200:205] = 3.0                         # one real spike
    s = np.arange(n) * 0.3                   # arc length, m
    idx = mad_spikes(h, s, min_sep_m=2.0, smooth_w=9, z_thresh=3.0)
    assert any(195 <= i <= 210 for i in idx), idx
    assert (n - 1) not in idx                # the trace END is never a node
    # with mean/std the transient inflates the threshold; the spike still
    # clears it here, but the two centres must give the same spike
    idx_mean = mad_spikes(h, s, 2.0, 9, 3.0, centre="mean")
    assert any(195 <= i <= 210 for i in idx_mean)


def test_mad_spikes_can_return_nothing():
    # UNIFORM jitter: bounded, so z can never reach 3 sigma (Gaussian tails would)
    h = 0.8 + np.random.default_rng(0).uniform(-1e-3, 1e-3, 300)
    s = np.arange(300) * 0.3
    assert len(mad_spikes(h, s, 2.0, 9, 3.0)) == 0


def test_place_routes_madent_and_rejects_sgm_without_positions():
    h = 0.5 + np.random.default_rng(1).uniform(-0.01, 0.01, 100); h[50] = 5.0
    s = np.arange(100) * 1.0
    idx, _ = place("madent", s=s, entropy=h, min_sep_m=2.0, smooth_w=1)
    assert list(idx) == [50]
    try:
        place("sgm", dist=np.zeros((3, 3)))
    except ValueError as e:
        assert "path_xy" in str(e)
    else:
        raise AssertionError("sgm must require path_xy")
