"""Privileged critic (user 2026-11-07, "lets try privileged critic").

The offline critic-slope diagnosis (scripts/offline_critic_slope.py) found the
warm-up critic crediting the expert's braking approach for the +100 arrival:
it cannot see how far the goal is, so the actions that always come just before
the bonus look like what earns it. Cut the approach out of its data and the
tilt vanishes. The fix is to let the CRITIC see the route distance to the goal
-- state columns the critic reads and the actor (and the GRU that feeds it)
never does -- so braking near the goal is explained by the distance, and the
policy's own inputs stay exactly as they were.

`actor_state_cols` names the state columns the actor reads (None = all, every
earlier run). The critic keeps reading `goal_cols`.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import MODE_IDX, STATE, _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402

PRIV = (5, 6)                                         # the fixture's privileged columns
ACTOR_COLS = tuple(c for c in range(STATE) if c not in PRIV)


def _agent(**kw):
    return GatedRecurrentSAC(_cfg(actor_state_cols=ACTOR_COLS,
                                  goal_cols=tuple(range(STATE)), **kw))


def test_the_actor_never_reads_the_privileged_columns():
    agent = _agent()
    H = agent.cfg.hidden
    x = torch.randn(4, H + STATE)
    y = x.clone()
    y[:, H + 5] += 3.0
    y[:, H + 6] -= 2.0
    m1, s1 = agent._actor_forward(x)
    m2, s2 = agent._actor_forward(y)
    assert torch.equal(m1, m2) and torch.equal(s1, s2)
    z = x.clone()
    z[:, H + 0] += 3.0                                 # a column the actor does read
    assert not torch.equal(agent._actor_forward(z)[0], m1)


def test_the_critic_reads_the_privileged_columns():
    agent = _agent()
    h = torch.randn(4, agent.cfg.hidden)
    st = torch.randn(4, STATE)
    a = torch.rand(4, 2) * 2 - 1
    q1 = agent._q(agent.critics_g, h, st, agent.goal_cols, a)
    st2 = st.clone()
    st2[:, 5] += 3.0
    assert not torch.allclose(q1, agent._q(agent.critics_g, h, st2, agent.goal_cols, a))


def test_the_actor_is_narrower_and_checkpoints_keep_their_keys():
    agent = _agent()
    assert agent.actor.trunk[0].in_features == agent.cfg.hidden + len(ACTOR_COLS)
    plain = GatedRecurrentSAC(_cfg(goal_cols=tuple(range(STATE))))
    assert set(agent.actor.state_dict()) == set(plain.actor.state_dict())


def test_an_update_and_a_save_load_round_trip_work(tmp_path):
    agent = _agent()
    agent.update(_chunk(agent.cfg, [1, 0]))
    agent.save(str(tmp_path / "ck"))
    again = _agent()
    again.load(str(tmp_path / "ck"))
    x = torch.randn(3, agent.cfg.hidden + STATE)
    assert torch.equal(agent._actor_forward(x)[0], again._actor_forward(x)[0])


def test_the_actor_must_read_the_mode_bit():
    with pytest.raises(ValueError, match="mode bit"):
        _cfg(actor_state_cols=(0, 1, 2), goal_cols=tuple(range(STATE)))


def test_actor_state_cols_stay_inside_the_state():
    with pytest.raises(ValueError, match="outside"):
        _cfg(actor_state_cols=(0, MODE_IDX, STATE + 3), goal_cols=tuple(range(STATE)))


def test_the_gru_never_reads_a_column_the_actor_may_not():
    """The GRU's hidden state is the actor's input: a privileged column in
    core_state_cols would reach the policy through its memory."""
    with pytest.raises(ValueError, match="core_state_cols"):
        _cfg(actor_state_cols=ACTOR_COLS, core_state_cols=(1, 5),
             goal_cols=tuple(range(STATE)))


def test_a_loaded_checkpoint_keeps_the_configured_learning_rates(tmp_path):
    """User 2026-11-07, "increase temperature learning rate": run L warm-starts
    from c0 with ALPHA_LR=1e-2. torch's Optimizer.load_state_dict restores the
    SAVED param-group hyperparameters, so without a re-apply the checkpoint's
    3e-4 would silently replace the configured rate and run L would be run K."""
    saved = GatedRecurrentSAC(_cfg(lr=3e-4, alpha_lr=0.0, goal_cols=tuple(range(STATE))))
    saved.save(str(tmp_path / "ck"))
    agent = GatedRecurrentSAC(_cfg(lr=5e-4, alpha_lr=1e-2, goal_cols=tuple(range(STATE))))
    agent.load(str(tmp_path / "ck"))
    assert agent.opt_temp_g.param_groups[0]["lr"] == 1e-2
    assert agent.opt_temp_e.param_groups[0]["lr"] == 1e-2
    assert agent.opt_actor.param_groups[0]["lr"] == 5e-4
    assert agent.opt_critic.param_groups[0]["lr"] == 5e-4
