"""The 2-mode eval's per-pass episode counts (plan task 15).

User decision 2026-09-20 (binding): the eval runs 6 goal-mode episodes and 4
explore-mode episodes -- but THROUGH the new --episodes-goal /
--episodes-explore overrides, never by changing --episodes' default, so a
bare --episodes invocation keeps driving both passes with the same count it
always did. resolve_episode_counts is that contract in one pure function,
and it must treat an explicit 0 as an override (is not None, not truthiness).

WHY THE MODULE IS IMPORTED WITH STUBS. Importing
scripts/image_goal_2mode_eval.py the normal way would run its module-level
env block, which pins the arm's flag set into os.environ (OBS_WITH_STATE=1
among them -- the default elsewhere is 0) and loads the run's snapshot from
results/run_env/. pytest imports every test module at collection, so those
pins would leak into the import-time env reads of any test module collected
after this one. The eval's heavy imports (train_carla_tqc, eval_policy,
vae_nav.latent_source, vae_nav.policy_io) are therefore stubbed in
sys.modules for the import, os.environ is snapshotted before and restored
after, and only the pure helper is exercised. No CARLA anywhere.
"""
import importlib.util
import os
import pathlib
import sys
import types

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "image_goal_2mode_eval.py"

# Only the names the eval module binds at import time; none of the real
# modules is loaded, so nothing below jax/gym is touched.
_STUB_ATTRS = {
    "train_carla_tqc": ("make_env", "build_agent"),
    "eval_policy": ("run_episodes",),
    "vae_nav.latent_source": ("LatentSource", "from_env"),
    "vae_nav.policy_io": ("load_agent_state", "param_fingerprint"),
}


def _import_with_stubs():
    env = os.environ.copy()
    saved = {name: sys.modules.get(name) for name in _STUB_ATTRS}
    try:
        for name, attrs in _STUB_ATTRS.items():
            mod = types.ModuleType(name)
            for attr in attrs:
                setattr(mod, attr, object())
            sys.modules[name] = mod
        spec = importlib.util.spec_from_file_location("two_mode_eval",
                                                      _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.clear()
        os.environ.update(env)
        for name, saved_mod in saved.items():
            if saved_mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_mod
    return mod


ev = _import_with_stubs()


def _ns(episodes, goal=None, explore=None):
    return types.SimpleNamespace(episodes=episodes, episodes_goal=goal,
                                 episodes_explore=explore)


def test_default_is_unchanged_both_passes():
    # No overrides: exactly the old behaviour, args.episodes on both passes.
    assert ev.resolve_episode_counts(_ns(25)) == (25, 25)


def test_both_overrides_apply_per_pass():
    # The user's binding 6 / 4 split, with --episodes still at its default.
    assert ev.resolve_episode_counts(_ns(25, goal=6, explore=4)) == (6, 4)


def test_one_side_overridden_other_falls_back_to_default():
    assert ev.resolve_episode_counts(_ns(7, goal=6)) == (6, 7)
    assert ev.resolve_episode_counts(_ns(7, explore=4)) == (7, 4)


def test_explicit_zero_is_an_override_not_an_unset_flag():
    # The `or` mistake would turn a requested 0-episode pass into the
    # default count; the contract is "override if given".
    assert ev.resolve_episode_counts(_ns(25, goal=0, explore=0)) == (0, 0)


def test_helper_is_pure_over_a_plain_namespace():
    # Anything argparse.Namespace-shaped works -- the helper reads exactly
    # the three attrs and mutates nothing.
    import argparse

    ns = argparse.Namespace(episodes=9, episodes_goal=3,
                            episodes_explore=None)
    before = vars(ns).copy()
    assert ev.resolve_episode_counts(ns) == (3, 9)
    assert vars(ns) == before