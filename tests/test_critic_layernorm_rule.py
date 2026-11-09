"""USER RULE 2026-11-06: "MAKE IT A RULE TO USE LAYER NORM IN CRITIC".

Critic LayerNorm bounds how high the critic extrapolates for actions the data
never covered -- the failure behind the collapsed SAC fine-tunes. It is the
trainer's default and a training run refuses to start without it. Checkpoints
made before the rule have no-LN critics, so: an eval builds the critic the
checkpoint has (config.json), and a warm start or a clone init from one gets
fresh LN critics (a clone's critics were never trained).
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_critic_layernorm_is_the_default():
    assert 'critic_layer_norm=(os.environ.get("CRITIC_LAYERNORM", "1") == "1"' in SRC


def test_a_training_run_refuses_critic_layernorm_off():
    from train_carla_sac import require_critic_layernorm
    with pytest.raises(SystemExit):
        require_critic_layernorm(False, training=True)
    require_critic_layernorm(True, training=True)
    require_critic_layernorm(False, training=False)       # evals of old checkpoints
    main_src = SRC[SRC.index("\ndef main():"):]
    assert "require_critic_layernorm(" in main_src


def test_the_checkpoints_own_critic_setting_is_read(tmp_path):
    from train_carla_sac import checkpoint_critic_layer_norm
    for val, want in ((True, True), (False, False)):
        d = tmp_path / str(val)
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"critic_layer_norm": val}))
        assert checkpoint_critic_layer_norm(str(d)) is want
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({"hidden": 16}))   # before the field
    assert checkpoint_critic_layer_norm(str(old)) is False


def test_evals_build_the_critic_the_checkpoint_has():
    main_src = SRC[SRC.index("\ndef main():"):]
    assert ("build_agent(binoc_dim, ACT_DIM, critic_layer_norm="
            "checkpoint_critic_layer_norm(EVAL_FROM) if EVAL_FROM else None") in main_src
    # and its critic input head (tests/test_critic_projection.py)
    assert "critic_proj_dim=checkpoint_critic_proj_dim(EVAL_FROM) if EVAL_FROM else None" in main_src


def test_a_warm_start_from_a_no_layernorm_checkpoint_gets_fresh_layernorm_critics(tmp_path):
    old = GatedRecurrentSAC(_cfg(critic_layer_norm=False))
    old.save(str(tmp_path / "clone"))
    agent = GatedRecurrentSAC(_cfg(critic_layer_norm=True))
    with pytest.raises(RuntimeError):                        # strict, as before
        agent.load(str(tmp_path / "clone"))
    agent = GatedRecurrentSAC(_cfg(critic_layer_norm=True))
    agent.load(str(tmp_path / "clone"), fresh_critics_on_mismatch=True)
    assert all(torch.equal(a, b) for a, b in zip(old.actor.parameters(), agent.actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(old.core.parameters(), agent.core.parameters()))
    assert agent.critics_g.layer_norm


def test_warm_starts_and_clone_inits_use_the_tolerant_load():
    assert "agent.load(INIT_FROM, fresh_critics_on_mismatch=True)" in SRC
    bc = open(os.path.join(ROOT, "torch_rl", "bc_trainer.py")).read()
    assert "agent.load(init_from, fresh_critics_on_mismatch=True)" in bc
