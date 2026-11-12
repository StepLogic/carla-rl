"""The Q-vs-fault analyser on hand-built traces with known answers."""
import numpy as np

from vae_nav.q_fault_analysis import analyse, auroc, per_step_arrays


def _row(ep, t, q, r=-1.0, fault=False, term=False, rem=None, crossing=None):
    return dict(ep=ep, t=t, q=q, r=r, term=term, trunc=False, off_road=fault,
                collision=False, crossing=crossing, arrived=False,
                remaining_m=rem, speed_kmh=10.0)


def test_auroc_known_values():
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert auroc([1, 2], [0, 0]) is None


def test_time_to_fault_and_return_to_go():
    rows = [_row(0, 0, -3.0), _row(0, 1, -2.0), _row(0, 2, -1.0, fault=True),
            _row(0, 3, 0.0, r=10.0, term=True)]
    a = per_step_arrays(rows, gamma=0.5)
    assert list(a["ttf"]) == [2, 1, 0, np.inf]
    # G3 = 10; G2 = -1 + 5 = 4; G1 = -1 + 2 = 1; G0 = -1 + 0.5 = -0.5
    assert np.allclose(a["g"], [-0.5, 1.0, 4.0, 10.0])
    assert a["term"].all()


def test_warning_critic_scores_high_and_blind_one_half():
    rng = np.random.default_rng(0)
    warn, blind = [], []
    for e in range(40):
        f = int(rng.integers(8, 30))
        for t in range(40):
            d = f - t
            fault = t == f
            q_w = -5.0 - (20 - 2 * d if 0 <= d < 10 else 0.0)
            warn.append(_row(e, t, q_w, fault=fault, rem=100.0 - t))
            blind.append(_row(e, t, float(rng.normal()), fault=fault, rem=100.0 - t))
    w = analyse(warn, gamma=0.99)["anticipation"]["fault_within_5"]
    b = analyse(blind, gamma=0.99)["anticipation"]["fault_within_5"]
    assert w["auroc_raw"] > 0.95 and w["auroc_within_distance_bins"] > 0.95
    assert abs(b["auroc_raw"] - 0.5) < 0.08


def test_distance_confound_removed_by_bins():
    # Q tracks distance only; faults happen only far from the goal. Raw AUROC
    # credits the critic, the within-bin AUROC must not.
    rows = []
    for e in range(60):
        far = e % 2 == 0
        base = 150.0 if far else 10.0
        for t in range(30):
            rem = base            # Q is a function of distance alone
            fault = far and t % 7 == 3
            rows.append(_row(e, t, -rem, fault=fault, rem=rem))
    a = analyse(rows, gamma=0.99)["anticipation"]["fault_within_5"]
    assert a["auroc_raw"] > 0.8
    assert abs(a["auroc_within_distance_bins"] - 0.5) < 0.15


def test_serious_ignores_legal_crossings():
    rows = [_row(0, 0, -1.0), _row(0, 1, -1.0, crossing="junction"),
            _row(0, 2, -1.0, crossing="illegal", term=True)]
    assert list(per_step_arrays(rows, 0.99)["ttf"]) == [1, 0, 0]
    assert list(per_step_arrays(rows, 0.99, serious=True)["ttf"]) == [2, 1, 0]


def test_truncated_episode_is_not_calibrated():
    rows = [_row(0, 0, -1.0), dict(_row(0, 1, -1.0, term=True), trunc=True)]
    assert not per_step_arrays(rows, 0.99)["term"].any()
