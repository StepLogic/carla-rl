"""N-step returns: segment assembly, boundary cuts, and the gamma**n' target.

WHAT N-STEP BUYS AND WHAT IT DOES NOT. The TD fixed point is the
gamma-discounted return regardless of n -- n-step does NOT fix the goal
critic's far-field saturation under gamma=0.99 (that is structural, horizon
1/(1-gamma) ~ 100 steps). What it changes is the PATH: the arrival bonus
propagates n steps per update instead of one, and TQC's per-step pessimism
compounds n times fewer times. The user picked it for exactly that, knowing
the distinction.

THE QUIET FAILURE THIS FILE GUARDS. Rows of one segment are NOT contiguous in
storage -- two envs insert interleaved and HER flushes splice whole blocks
between an episode's steps -- so a merge that walks forward by storage index
would chain the WRONG rows, silently, and every reward would still look
plausible. The buffer therefore chains by per-slot successor pointers stamped
with generations; the tests below would catch a regression to storage
adjacency, and so would the ring tests.

N_STEP=1 MUST BE BYTE-IDENTICAL to the old buffer: no `n_steps` field, raw
rewards, the plain jaxrl2 sample path.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
pytest.importorskip("jaxrl2")
gym = pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402

from vae_nav.mode_split_buffer import ModeSplitBuffer  # noqa: E402

GOAL, EXPLORE = 1, 0
GAMMA = 0.9


def _spaces():
    return (spaces.Dict({"latent": spaces.Box(-1, 1, (4,), np.float32),
                         "vector": spaces.Box(-1, 1, (3,), np.float32)}),
            spaces.Box(-1, 1, (2,), np.float32))


def _row(tag, reward=-1.0, mask=1.0, done=0.0):
    """A goal-mode row whose latent carries its STEP INDEX (next = tag + 1),
    so every sampled merged row can be attributed to the segment position it
    started at AND its bootstrap state can be checked."""
    o = {"latent": np.full(4, tag, np.float32),
         "vector": np.full(3, tag, np.float32)}
    no = {"latent": np.full(4, tag + 1, np.float32),
          "vector": np.full(3, tag, np.float32)}
    return (dict(observations=o, actions=np.zeros(2, np.float32),
                 rewards=np.float32(reward), masks=np.float32(mask),
                 dones=np.float32(done), next_observations=no),
            GOAL)


def _fill_episode(buf, n, seg, arrival=True):
    """n rows of one segment, -1 per step, arriving on the last row."""
    for t in range(n):
        last = arrival and t == n - 1
        row, mode = _row(t, reward=(10.0 if last else -1.0),
                         mask=(0.0 if last else 1.0),
                         done=(1.0 if last else 0.0))
        buf.insert(row, mode=mode, segment=seg)


# -- assembly ----------------------------------------------------------------

def test_merged_row_is_the_discounted_sum_with_the_n_prime_bootstrap():
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, n_step=3,
                          discount_goal=GAMMA, discount_explore=GAMMA)
    _fill_episode(buf, 10, seg=1)

    batch = dict(buf.sample(64))
    starts = np.asarray(batch["observations"]["latent"])[:, 0]
    for t, r, m, d, nprime, nxt in zip(
            starts, batch["rewards"], batch["masks"], batch["dones"],
            batch["n_steps"], batch["next_observations"]["latent"][:, 0]):
        t = int(t)
        if t <= 6:      # rows t, t+1, t+2, none terminal
            n_want, m_want, d_want = 3, 1.0, 0.0
            r_want = -(1.0 + GAMMA + GAMMA ** 2)
        elif t == 7:    # rows 7, 8, 9 -- row 9 ARRIVES and is included
            n_want, m_want, d_want = 3, 0.0, 1.0
            r_want = -1.0 + GAMMA * -1.0 + GAMMA ** 2 * 10.0
        elif t == 8:
            n_want, m_want, d_want = 2, 0.0, 1.0
            r_want = -1.0 + GAMMA * 10.0
        else:
            n_want, m_want, d_want = 1, 0.0, 1.0
            r_want = 10.0
        assert float(nprime) == n_want, (t, nprime)
        assert float(r) == pytest.approx(r_want), (t, r)
        assert float(m) == m_want and float(d) == d_want, (t, m, d)
        # THE BOOTSTRAP STATE is the successor of the last merged row.
        assert float(nxt) == t + n_want, (t, nxt)


def test_a_segment_boundary_cuts_the_chain():
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, n_step=3,
                          discount_goal=GAMMA, discount_explore=GAMMA)
    for t in range(5):
        buf.insert(_row(t)[0], mode=GOAL, segment=1)
    for t in range(5, 10):
        buf.insert(_row(t)[0], mode=GOAL, segment=2)

    batch = dict(buf.sample(64))
    starts = np.asarray(batch["observations"]["latent"])[:, 0]
    for t, nprime, nxt in zip(starts, batch["n_steps"],
                              batch["next_observations"]["latent"][:, 0]):
        t = int(t)
        n_want = min(3, 5 - t) if t < 5 else min(3, 10 - t)
        assert float(nprime) == n_want, (t, nprime)
        assert float(nxt) == t + n_want, (t, nxt)
        # A cut segment bootstraps ON (mask 1): the truncated row's own
        # convention, carried to the merged row.
        assert float(batch["masks"][list(starts).index(t)]) == 1.0


def test_ring_overwrite_never_chains_a_stale_slot():
    """Capacity 5 per mode, 14 rows of ONE segment. Rows t <= 7 point at
    slots whose content has since moved five rows on -- the generation check
    must stop the chain there, or the reward sum would mix rows from a
    different pass through the ring."""
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 10, n_step=3,
                          discount_goal=GAMMA, discount_explore=GAMMA)
    for t in range(14):
        buf.insert(_row(t)[0], mode=GOAL, segment=1)

    batch = dict(buf.sample(64))
    starts = np.asarray(batch["observations"]["latent"])[:, 0]
    for t, nprime in zip(starts, batch["n_steps"]):
        t = int(t)
        if t <= 7:
            n_want = 1     # successor slot now holds a much later row
        elif t <= 11:
            n_want = 3     # rows t, t+1, t+2 all still resident
        elif t == 12:
            n_want = 2     # row 14 never existed
        else:
            n_want = 1
        assert float(nprime) == n_want, (t, nprime)


def test_n_step_one_is_byte_identical_to_the_old_path():
    """No `n_steps` field, unmerged rewards, and the plain jaxrl2 draw."""
    obs_sp, act_sp = _spaces()

    def build():
        b = ModeSplitBuffer(obs_sp, act_sp, 400)
        for m in (1, 0):
            b._buf[m].seed(1234)
        _fill_episode(b, 10, seg=1)
        return b

    a, b = build(), build()
    for m in (1, 0):
        a._buf[m].seed(1234)
        b._buf[m].seed(1234)
    ba, bb = a.sample(32), b.sample(32)
    assert "n_steps" not in ba.keys()
    for k in ba.keys():
        assert np.array_equal(np.asarray(ba[k]), np.asarray(bb[k])), k
    # Raw per-step rewards, never merged (the +10 arrival may appear; nothing
    # is ever summed).
    r = np.asarray(ba["rewards"])
    assert set(np.unique(r).tolist()) <= {-1.0, 10.0}, np.unique(r)


def test_two_envs_with_colliding_counters_never_merge():
    """The trap the segment tag exists for: two per-env relabel buffers both
    counting from 1 over one episode each. With the env index tagged into the
    high bits the two chains can never cross -- untagged, the ids would alias
    and env 1's rows would be summed onto env 0's episode."""
    obs_sp, act_sp = _spaces()
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, n_step=3,
                          discount_goal=GAMMA, discount_explore=GAMMA)
    for t in range(4):                      # env 0, one episode, segment 1
        buf.insert(_row(t)[0], mode=GOAL, segment=1)
    for t in range(4):                      # env 1, tagged segment
        buf.insert(_row(100 + t)[0], mode=GOAL, segment=(1 << 32) + 1)

    batch = dict(buf.sample(64))
    starts = np.asarray(batch["observations"]["latent"])[:, 0]
    for t, nprime in zip(starts, batch["n_steps"]):
        t = int(t)
        local = t if t < 100 else t - 100
        assert float(nprime) == min(3, 4 - local), (t, nprime)


# -- the learner's gamma**n' bootstrap ----------------------------------------

def test_tqc_target_accepts_a_per_row_discount():
    """The learner's per-row discount is passed PRE-SHAPED to (batch, 1),
    because the vendored tqc_target computes `discount * masks[:, None]`:
    a FLAT (batch,) discount broadcasts there as (1, batch) x (batch, 1) ->
    (batch, batch) -- a silent outer product that happens to be an instant
    shape error against soft (batch, kept), or quietly WRONG MATH when
    kept == batch. (batch, 1) x (batch, 1) -> (batch, 1) x (batch, kept)
    is column-wise, exactly like the scalar. Pinning this here means
    jaxrl2 stays untouched and the contract is a fact, not an assumption."""
    from jaxrl2.agents.tqc.quantiles import tqc_target, truncated_targets

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (2, 6, 5), jnp.float32) * 3.0
    rewards = jnp.asarray([-1.0, -1.0, -1.0, 10.0, -1.0, -1.0])
    masks = jnp.asarray([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    alpha = jnp.float32(0.2)
    nlp = jnp.zeros(6, jnp.float32)

    scalar = tqc_target(q, rewards, masks, discount=0.9,
                        top_quantiles_to_drop=1, alpha=alpha, next_log_probs=nlp)
    per_row = tqc_target(q, rewards, masks,
                         discount=jnp.full((6, 1), 0.9, jnp.float32),
                         top_quantiles_to_drop=1, alpha=alpha, next_log_probs=nlp)
    assert jax.numpy.allclose(scalar, per_row, atol=1e-6)

    # And gamma**n' with n'=2 really is 0.81 on the rows that say 2.
    discounts = jnp.asarray([[0.9], [0.9], [0.81], [0.9], [0.9], [0.9]])
    got = tqc_target(q, rewards, masks, discount=discounts,
                     top_quantiles_to_drop=1, alpha=alpha, next_log_probs=nlp)
    kept = truncated_targets(q, 1)
    want = rewards[:, None] + discounts * masks[:, None] * kept
    assert jax.numpy.allclose(got, want, atol=1e-5)

    # THE TRAP ITSELF, pinned at the broadcast level -- a flat discount fed to
    # tqc_target would raise inside the vendored code ((6,6) vs (6, kept)),
    # which is this exact multiplication blowing up.
    flat = jnp.full(6, 0.9, jnp.float32)
    assert (flat * masks[:, None]).shape == (6, 6)          # outer product
    assert (discounts * masks[:, None]).shape == (6, 1)     # the fix, column-wise


def test_a_real_n_step_batch_drives_a_real_update():
    """The gated learner consumes the assembled batch (extra `n_steps` key,
    per-row discount) and both critics stay finite."""
    from vae_nav.gated_tqc_learner import GatedTQCLearner

    obs_sp = spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
        "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
        "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
    act_sp = spaces.Box(np.array([-0.5, 0.0], np.float32),
                        np.array([0.5, 0.6], np.float32))
    agent = GatedTQCLearner(
        seed=0, observations=obs_sp.sample(), actions=act_sp.sample(),
        discount=0.99, discount_e=0.99, hidden_dims=(32, 32),
        num_qs=2, num_quantiles=5, top_quantiles_to_drop_per_net=1,
        target_entropy=-2.0, action_space=act_sp)
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, n_step=3,
                          discount_goal=agent.discount,
                          discount_explore=agent.discount_e)
    rng = np.random.default_rng(0)
    for mode in (GOAL, EXPLORE):
        for t in range(60):
            o = {k: rng.normal(size=s.shape).astype(np.float32)
                 for k, s in obs_sp.spaces.items()}
            o["mode"] = np.array([float(mode)], np.float32)
            no = dict(o)
            buf.insert(dict(observations=o,
                            actions=rng.uniform(-0.5, 0.5, 2).astype(np.float32),
                            rewards=np.float32(-1.0), masks=np.float32(1.0),
                            dones=np.float32(0.0), next_observations=no),
                       mode=mode, segment=(mode << 20) + t // 10)
    batch = buf.sample(16)
    assert "n_steps" in batch.keys()
    for _ in range(3):
        info = agent.update(batch)
    for k, v in info.items():
        assert np.isfinite(float(v)), (k, v)


# -- through the HER pipeline --------------------------------------------------

STEP_COST = 1.0
BONUS = 10.0
FINE = 100.0


@pytest.fixture
def counting(monkeypatch):
    from vae_nav import gated_reward as gr
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", True)
    monkeypatch.setattr(gr, "STEP_COST", STEP_COST)
    monkeypatch.setattr(gr, "GOAL_ARRIVAL_BONUS", BONUS)
    monkeypatch.setattr(gr, "GOAL_OFF_ROAD_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_LANE_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_COLLISION_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_OFFROAD_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_COLLISION_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_CROSSING_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_MAX_STEPS", 1000)
    monkeypatch.setattr(gr, "GOAL_SPEED_BONUS", 0.0)


def _her_block(counting):
    """A real HER buffer over a real n-step mode-split buffer.

    goal_frac=1.0 sends the WHOLE sample to the goal side, where every
    hindsight row lives -- the on-policy rows are explore rows (mode=0) and
    stay out of the way. The obs space must match the drive() txns EXACTLY
    (goal 2-d + latent 4-d): a real ReplayBuffer asserts row keys == dataset
    keys, so a mismatched fixture dies in _insert_recursively, not in the
    logic under test.
    """
    from vae_nav.episode_her import EpisodeRelabelBuffer

    obs_sp = spaces.Dict({"goal": spaces.Box(-1e3, 1e3, (2,), np.float32),
                          "latent": spaces.Box(-1e3, 1e3, (4,), np.float32),
                          "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
    act_sp = spaces.Box(-1, 1, (2,), np.float32)
    buf = ModeSplitBuffer(obs_sp, act_sp, 400, goal_frac=1.0, n_step=3,
                          discount_goal=GAMMA, discount_explore=GAMMA)
    her = EpisodeRelabelBuffer(
        buf, lambda fields: np.asarray(fields["goal_pos"], np.float32)
        - np.asarray(fields["pos"], np.float32),
        goal_radius=2.0, step_cost=STEP_COST, time_to_goal=True, gated=True,
        recost_counting=True, k_goal=600.0, k_collision=5.0, k_lane=5.0,
        k_living=0.5, k_dist=1.0, seed=0, segment_tag=0)
    return her, buf


def _drive(her, n=12):
    for t in range(n):
        pos = (float(t), 0.0)
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32),
                                "mode": np.zeros(1, np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32),
                                     "mode": np.zeros(1, np.float32)},
               "actions": np.zeros(2, np.float32),
               "rewards": np.float32(-STEP_COST),
               "masks": np.float32(1.0),
               "dones": np.float32(0.0)}
        her.record(txn, {"pos": pos}, pos, hit=False,
                   speed_kmh=22.0, target_speed_kmh=20.0, mode=EXPLORE)
    her.flush_episode()


def test_her_rows_chain_within_a_shared_hindsight_goal(counting):
    """Every relabelled row drawn to tp=6 shares one segment. The merged
    reward at start t is the discounted sum of the recosted counting-branch
    rewards, the arrival row's +bonus lands at power gamma**(n'-1), and the
    merged masks/dones are the arrival row's."""
    her, buf = _her_block(counting)
    her._pick_goal = lambda i, n, pos: (6 if i < 6 else min(i + 1, n - 1))
    _drive(her, 12)
    assert her.last_pairs, "no hindsight rows produced; the test proves nothing"

    batch = dict(buf.sample(64))
    starts = np.asarray(batch["observations"]["latent"])[:, 0]
    assert len(set(starts.tolist())) > 4, "with-replacement draw degenerated"
    for t, r, m, d, nprime, nxt, goal_obs in zip(
            starts, batch["rewards"], batch["masks"], batch["dones"],
            batch["n_steps"], batch["next_observations"]["latent"][:, 0],
            batch["observations"]["goal"]):
        t = int(t)
        if not np.all(goal_obs == 0):
            # Rows 0..5 share tp=6: rows 4 and 5 ARRIVE (|pos[i+1]-pos[6]| < 2),
            # everything else pays -1 and bootstraps on. Rows 6..10 draw tp=i+1,
            # which every row arrives at immediately -- arrival singletons.
            if t <= 1:                       # rows t, t+1, t+2, all clean
                n_want, m_want, d_want, r_want = 3, 1.0, 0.0, -(1 + GAMMA + GAMMA ** 2)
            elif t == 2:                     # rows 2, 3, 4 -- row 4 arrives
                n_want, m_want, d_want = 3, 0.0, 1.0
                r_want = -1.0 + GAMMA * -1.0 + GAMMA ** 2 * BONUS
            elif t == 3:
                n_want, m_want, d_want = 2, 0.0, 1.0
                r_want = -1.0 + GAMMA * BONUS
            else:                            # arrival rows, single or terminal
                n_want, m_want, d_want, r_want = 1, 0.0, 1.0, float(BONUS)
        else:
            # ON-POLICY EXPLORE ROW. goal_frac=1.0 does NOT give an
            # all-goal batch: sample() forces max(n_exp, 1) so each side
            # always gets at least one row, and drive() filed its on-policy
            # steps as explore rows under the episode's own segment. They
            # chain too -- asserting their profile makes this test cover the
            # on-policy path through the same HER pipeline.
            n_want = min(3, 12 - t)
            m_want, d_want = 1.0, 0.0
            r_want = -sum(GAMMA ** k for k in range(n_want))
        assert float(nprime) == n_want, (t, nprime, bool(np.all(goal_obs == 0)))
        assert float(r) == pytest.approx(r_want), (t, r)
        assert float(m) == m_want and float(d) == d_want, (t, m, d)
        assert float(nxt) == t + n_want, (t, nxt)


def test_a_hindsight_goal_change_starts_a_new_segment(counting):
    """tp varies per row in general -- _pick_goal draws independently -- so a
    tp change or a step gap must cut the chain rather than summing two
    different goals' rewards."""
    her, buf = _her_block(counting)
    her._pick_goal = lambda i, n, pos: min(i + 4, n - 1) if i + 4 <= n - 1 else None
    _drive(her, 12)
    # tp = i+4 differs for every pair: every hindsight row is a singleton run.
    assert her.last_pairs
    batch = dict(buf.sample(64))
    # Hindsight rows carry goal obs = pos[tp]-pos[i] = (4, 0), never zero; the
    # one forced explore row per batch (max(n_exp, 1)) carries goal zeros and
    # chains n'=min(3, 12-t) under the episode segment. Assert each population
    # against its own profile.
    goal_rows, exp_rows = [], []
    for row in zip(
            np.asarray(batch["observations"]["latent"])[:, 0],
            batch["rewards"], batch["masks"], batch["n_steps"],
            batch["observations"]["goal"]):
        (goal_rows if not np.all(row[4] == 0) else exp_rows).append(row)
    assert goal_rows, "no hindsight rows in the batch"
    for t, r, m, nprime, _ in goal_rows:
        assert float(nprime) == 1.0, (t, nprime)
        assert float(r) == pytest.approx(-1.0)   # d_next = 4 m > radius 2
        assert float(m) == 1.0
    for t, r, m, nprime, _ in exp_rows:
        n_want = min(3, 12 - int(t))
        assert float(nprime) == n_want, (t, nprime)
        assert float(r) == pytest.approx(-sum(GAMMA ** k for k in range(n_want)))
        assert float(m) == 1.0