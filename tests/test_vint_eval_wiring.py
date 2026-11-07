"""EVAL_POLICY=vint: run_eval drives ViNT (torch_rl.vint_driver) on c0's exact
episodes (user 2026-11-07: "evaluate" ViNT like c0). The episodes, the faults
and the scoring are run_eval's; only the action source changes."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_eval_policy_is_sac_or_vint():
    from train_carla_sac import validate_eval_policy
    assert validate_eval_policy("sac") == "sac" and validate_eval_policy("vint") == "vint"
    with pytest.raises(ValueError):
        validate_eval_policy("nomad")


def test_pixel_frames_are_what_the_recorder_stored():
    """run_record kept pxr.to_frame(info["debit_cur"], MONO_IMG) as uint8; ViNT
    learned from those, so the eval hands it the same thing, goal included."""
    from torch_rl import pixel_rows as pxr
    from train_carla_sac import pixel_frames
    info = {"debit_cur": np.full((64, 64, 3), 7, np.uint8),
            "debit_goal": np.full((256, 256, 3), 9, np.uint8)}
    cur, goal = pixel_frames(info, pxr)
    assert cur.shape == goal.shape == (pxr.MONO_IMG, pxr.MONO_IMG, 3)
    assert cur.dtype == goal.dtype == np.uint8
    assert int(cur[0, 0, 0]) == 7 and int(goal[0, 0, 0]) == 9
    assert 'pxr.to_frame(info["debit_cur"], pxr.MONO_IMG).astype(np.uint8)' in SRC


def test_main_builds_the_vint_driver_before_any_sac_agent():
    main_src = SRC[SRC.index("\ndef main():"):]
    i_vint = main_src.index('if EVAL_POLICY == "vint":')
    assert i_vint < main_src.index("agent = build_agent(")
    assert "load_vint_policy(EVAL_FROM" in main_src[i_vint:]
    # ViNT's distance is in steps of stride frames, 0.1 s each
    assert 'step_s=meta["stride"] * meta["frame_dt_s"]' in main_src[i_vint:]


def test_run_eval_takes_the_pixel_path_for_a_pixel_policy():
    body = SRC[SRC.index("\ndef run_eval("):SRC.index("\ndef coverage_perturbation(")]
    assert 'getattr(agent, "drives_from_pixels", False)' in body
    assert "agent.act_env(" in body and "pixel_frames(info, pxr)" in body


# ------------------------------------------------- map navigation (user 2026-11-07)
class _Gru:
    drives_from_pixels = False

    def __init__(self):
        self.resets = 0

    def reset_hidden(self, n):
        self.resets += 1
        return "fresh"


class _Pixels(_Gru):
    drives_from_pixels = True


def test_a_subgoal_switch_resets_c0s_gru_but_keeps_vints_frames():
    from train_carla_sac import subgoal_reset
    c0 = _Gru()
    assert subgoal_reset(c0, "old") == "fresh" and c0.resets == 1
    vint = _Pixels()
    assert subgoal_reset(vint, "old") == "old" and vint.resets == 0


def test_every_map_reset_in_run_eval_goes_through_subgoal_reset():
    body = SRC[SRC.index("\ndef run_eval("):SRC.index("\ndef coverage_perturbation(")]
    assert body.count("hidden = subgoal_reset(agent, hidden)") == 3
    assert "if EVAL_MAP_RESET_HIDDEN:\n                            hidden = agent.reset_hidden(1)" not in body
    assert "if EVAL_MAP_RESET_HIDDEN:\n                        hidden = agent.reset_hidden(1)" not in body


def test_vint_slows_only_for_the_final_goal_on_a_map():
    body = SRC[SRC.index("\ndef run_eval("):SRC.index("\ndef coverage_perturbation(")]
    i = body.index("agent.act_env(")
    assert "agent.final_goal = nav is None or bool(nav.final)" in body[i - 400:i]


def test_vint_topomap_is_wired_into_run_eval():
    """EVAL_VINT_TOPOMAP (user 2026-11-07: "ViNT-native topomap"): run_eval
    localises and plans per episode with VintTopoEpisode and lets it drive."""
    import train_carla_sac as m
    assert m.EVAL_VINT_TOPOMAP == os.environ.get("EVAL_VINT_TOPOMAP", "")
    body = SRC[SRC.index("\ndef run_eval("):SRC.index("\ndef coverage_perturbation(")]
    assert "VintTopoEpisode(topo_map, agent.policy)" in body
    assert "topo.start(*pixel_frames(info, pxr))" in body
    assert "topo.act(agent, *pixel_frames(info, pxr), _speed_kmh(info))" in body
    assert '"vint_topo"' in body and "topo.diagnostics(" in body
    assert '_vinttopo-' in body                     # its own output folder


def test_the_vint_topomap_and_the_benchmark_map_are_exclusive():
    assert ("if EVAL_VINT_TOPOMAP and (EVAL_MAP_DIR or EVAL_POLICY != \"vint\"):" in SRC)


def test_vint_may_run_map_evals():
    main_src = SRC[SRC.index("\ndef main():"):]
    i = main_src.index('if EVAL_POLICY == "vint":')
    block = main_src[i:main_src.index("agent = build_agent(", i)]
    assert "EVAL_MAP_DIR or" not in block
