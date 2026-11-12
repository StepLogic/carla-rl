"""An eval builds the critic ENSEMBLE its checkpoint has (2026-11-09).

Job 465314 trains standard SAC's twin critics (N_CRITICS=2). Its in-job evals load
each checkpoint strictly but built the default 10 critics -- N_CRITICS is the
training overlay's, not the eval's -- and failed ("size mismatch for proj_w ... [2,
3650, 50] ... [10, 3650, 50]"). Like critic_layer_norm and critic_proj_dim, the eval
now takes n_critics and subset_size from the checkpoint's config.json; a config
without them (older) falls back to the environment, as before. Checkpoints made with
10 critics (c0, every run before) record n_critics=10, so their evals are unchanged.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_the_checkpoints_ensemble_is_read(tmp_path):
    from train_carla_sac import checkpoint_critic_ensemble
    (tmp_path / "config.json").write_text(json.dumps({"n_critics": 2, "subset_size": 2}))
    assert checkpoint_critic_ensemble(str(tmp_path)) == {"n_critics": 2, "subset_size": 2}
    (tmp_path / "config.json").write_text(json.dumps({"hidden": 16}))   # before the fields
    assert checkpoint_critic_ensemble(str(tmp_path)) == {"n_critics": None, "subset_size": None}


def test_build_agent_takes_them_and_the_eval_passes_them():
    import inspect
    from train_carla_sac import build_agent
    params = inspect.signature(build_agent).parameters
    assert "n_critics" in params and "subset_size" in params
    assert 'env_int("N_CRITICS", 10) if n_critics is None else int(n_critics)' in SRC
    assert 'env_int("SUBSET_SIZE", 2) if subset_size is None else int(subset_size)' in SRC
    main_src = SRC[SRC.index("\ndef main():"):]
    assert "**(checkpoint_critic_ensemble(EVAL_FROM) if EVAL_FROM else {})" in main_src
