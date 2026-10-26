"""The training loop's wiring, checked at the source and against a fake env.

The loop itself needs CARLA and a GPU, so what is testable here is the wiring --
and the wiring is where this kind of script goes wrong silently. Each test names
a mistake that would still produce a running job and a plausible curve:

  * the env encoding the binocular vector as well as the learner (7 ms a step for
    a vector nothing reads)
  * the mode bit missing from, or misplaced in, the state block, so the learner's
    mode_index reads a velocity
  * the goal frame re-latched mid-episode, or not re-latched on reset
  * the previous action in the row not being the action that was actually taken
  * updates starting before the warm-up, or sampling a buffer that cannot yet
    fill a chunk
  * the log leading with the goal-arm rate instead of policy=
"""
import os
import re

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "train_carla_sac.py")


def src():
    with open(SRC) as fh:
        return fh.read()


# ------------------------------------------------------------ the encoder --
# ------------------------------------------------------------- the row --
def test_the_state_block_is_the_MODE_BIT_AND_NOTHING_ELSE():
    """User 2026-10-14: "get rid of all state vectors except mode bit". The
    actor used to receive its own last steer and throttle as observation -- a
    feedback path a policy can latch onto -- and the run that collapsed sat at
    full left lock for 800k steps."""
    s = src()
    assert 'pxr.mono_row(i["debit_cur"], o["mode"], o["binoc"])' in s
    assert 'o["vector"]' not in s and 'o["state"]' not in s
    assert "mode_index=0" in s


def test_mode_index_points_at_the_FIRST_state_column():
    """The state block is [mode | binoc], so the mode is column 0 and the binoc
    columns follow it. Off by one and the critic reads a binoc feature as the
    mode, which would not crash."""
    s = src()
    assert "mode_index=0" in s
    assert "mode_index=state_dim - 1" not in s     # the previous arm's layout
    assert "state_dim = pxr.mono_state_dim(binoc_dim)" in s


def test_the_single_critics_view_is_the_MODE_BIT():
    """User 2026-10-13: one critic for both reward modes, and the mode bit
    observed by it. Without that column the critic predicts two different
    returns from otherwise identical inputs and learns their average -- which
    does not fail, it just reads as mediocre. There is no other goal column:
    with goal_tokens the goal is an image the GRU has already folded in."""
    s = src()
    assert "one_critic=os.environ.get(\"ONE_CRITIC\", \"1\") == \"1\"" in s
    assert "goal_cols=tuple(range(state_dim)), explore_cols=()" in s
    assert "mode_index=0" in s                      # the same column


def test_there_is_NO_previous_action_anywhere():
    """User 2026-10-14: "remove prev action from obs space". Both feedback paths
    from the policy's output back into its input are now gone -- this one and
    the proprioceptive steer/throttle."""
    s = src()
    assert "prev_action=False" in s
    # NB "prev_a" alone also matches "prev_action=False", which is the flag that
    # turns the slot OFF -- so look for the variable's actual uses.
    assert "prev_a," not in s and "prev_a)" not in s and "prev_a =" not in s


def test_the_critic_sees_everything_the_actor_sees():
    """User 2026-10-14. The actor is handed the state block in full, so anything
    less for the critic makes it judge actions on strictly less information than
    the actor chose them with."""
    s = src()
    assert "goal_cols=tuple(range(state_dim))" in s


# ------------------------------------------------------------- the goal --
def test_the_goal_frame_is_latched_per_EPISODE():
    """Once at start and once per reset -- never per step, which would make the
    target drift under the policy."""
    s = src()
    # Counted in main() only: run_eval (EVAL_FROM) latches its own goal once
    # per eval episode, which is the same rule, in a different function.
    main_src = s[s.index("\ndef main():"):]
    assert main_src.count('pxr.goal_block(info["debit_goal"])') == 2
    body = s[s.index("for step in range(1, MAX_STEPS"):]
    assert 'goal_block(info["debit_goal"])' in body     # the reset branch
    assert body.count("goal_block(") == 1, "the goal must not be re-read per step"


def test_the_goal_goes_to_both_the_actor_and_the_buffer():
    s = src()
    assert "agent.act(row, hidden, goal=goal_blk)" in s
    assert "goal=goal_blk" in s[s.index("buf.add("):s.index("buf.add(") + 220]


def test_the_hidden_state_is_reset_with_the_episode():
    """A GRU carrying the last episode's memory into a new one is the classic
    recurrent off-policy bug, and it does not crash."""
    s = src()
    assert s.count("agent.reset_hidden(1)") >= 2


# ----------------------------------------------------------- the update --
def test_updates_wait_for_the_warm_up_and_for_a_full_chunk():
    s = src()
    assert "step > LEARNING_STARTS and step % UPDATE_EVERY == 0" in s
    assert "buf.sampleable >= CHUNK_T + BURN_IN + 1" in s


def test_random_actions_are_used_during_the_warm_up():
    s = src()
    assert "if step <= LEARNING_STARTS:" in s
    assert "env.action_space.sample()" in s


def test_the_update_cadence_default_is_the_measured_one():
    """209 ms per update against ~48 env steps/s: both flat out oversubscribe one
    GPU, so the default is one update per 20 steps."""
    s = src()
    assert 'env_int("UPDATE_EVERY", 20)' in s


def test_the_episode_is_ENDED_in_the_buffer_before_the_env_resets():
    """An episode left open is never sampleable, so the replay would fill with
    data the learner cannot use."""
    s = src()
    i_end, i_reset = s.index("buf.end_episode()"), s.index("obs, info = env.reset()")
    assert i_end < i_reset


# ------------------------------------------------------------ reporting --
def test_the_log_leads_with_policy():
    """User rule 2026-09-25: the arm serves two modes, and the goal-arm rate
    alone can read well while the explore half degrades."""
    s = src()
    line = s[s.index('f"[Ep '):s.index('f"[Ep ') + 200]
    assert line.index("policy=") < line.index("goal=")


def test_a_preflight_banner_is_printed_before_stepping():
    """User rule, after four restarts: label, checkpoint path, town, seeds and
    action mode in the run's own log."""
    s = src()
    i = s.index("preflight")
    j = s.index("for step in range(1, MAX_STEPS")
    assert i < j
    banner = s[i:j]
    assert "env.action_space" in s or "_box = env.action_space" in s, \
        "the banner must print the SPACE, not the env vars meant to shape it"
    for want in ("towns=", "seed=", "action box", "checkpoints ->", "reward:",
                 "HELD OUT", "FROZEN", "mode bit ONLY"):
        assert want in banner, want


def test_checkpoints_are_written_periodically_and_at_the_end():
    s = src()
    assert "step % CKPT_EVERY == 0" in s
    assert s.count("agent.save(d)") >= 2


# ------------------------------------------------------------- the videos --
def test_videos_record_the_camera_AND_the_goal_frame():
    """User 2026-10-14: videos for debugging. The pair is the point -- a camera
    panel alone cannot show whether the car is driving towards the view it was
    given, which is this arm's whole question."""
    s = src()
    assert 'vid.add(info["debit_cur"], info["debit_goal"]' in s


def test_only_five_episodes_are_kept():
    """User 2026-10-14: "Store only 5 episodes at a time"."""
    s = src()
    assert 'env_int("VIDEO_KEEP", 5)' in s


def test_a_video_is_started_for_the_first_episode_and_after_every_reset():
    """Missing the first means the run's opening episode is never recorded;
    missing the reset means every later video is the first episode again."""
    s = src()
    assert "vid.start(0)" in s
    assert "vid.start(episodes)" in s


def test_the_video_is_saved_with_the_end_reason():
    """So the interesting ones can be found without opening them: crossings and
    arrivals are different debugging sessions."""
    s = src()
    assert "vid.save(end)" in s
    i_end = s.index('end = str(info.get(_end_key')
    assert s.index("vid.save(end)") > i_end


def test_recording_happens_before_the_observation_is_advanced():
    """info must still be the step that was just acted on. Record after `obs,
    info = nobs, ninfo` and every frame is one step late -- the stale-frame bug
    that cost an earlier run its perception."""
    s = src()
    assert s.index('vid.add(info["debit_cur"]') < s.index("obs, info, row = nobs")


# --------------------------------------- the binocular + monocular stack --
def test_debit_is_FROZEN_and_no_adapters_are_added():
    """User 2026-10-14: "keep debit frozen". Frozen means its output is a fixed
    function of (view, goal), which is what lets the ENV precompute it instead
    of the learner re-encoding the chunk twice per update at 209 ms."""
    s = src()
    assert "add_encoder_adapters" not in s
    assert "debit_adapters" not in s


def test_the_env_IS_given_the_encoder_now():
    """The opposite of the previous arm: there the learner encoded, so the
    wrapper must not; here the encoder is frozen, so the wrapper encodes once
    per step and the learner never touches it."""
    s = src()
    assert "encoder=enc" in s
    assert "encoder=None" not in s, \
        "a None encoder makes the wrapper serve a ZERO binoc"


def test_the_trainable_visual_path_is_the_resnet():
    s = src()
    assert 'mono_encoder="resnet"' in s
    assert "grid=pxr.MONO_IMG" in s and "goal_tokens=False" in s


def test_the_binoc_reaches_the_GRU_through_core_state_cols():
    """The peer's own vc3 layout: the goal arrives already encoded, so it enters
    as state columns rather than as a stacked goal image."""
    s = src()
    assert "core_state_cols=pxr.mono_core_state_cols(binoc_dim)" in s


def test_an_unencoded_binoc_is_REFUSED():
    """A zero binoc trains the policy goal-blind and nothing downstream looks
    wrong -- the run simply never learns to use the goal."""
    s = src()
    assert 'i.get("binoc_encoded", False)' in s
    assert "RuntimeError" in s


def test_the_towns_exclude_town02_and_town03():
    """User 2026-10-14: all towns except those two, which stay held out."""
    s = src()
    i = s.index("ENV_TOWNS")
    line = s[i:i + 160]
    for t in ("Town01", "Town04", "Town05", "Town06", "Town07", "Town10HD"):
        assert t in line, t
    assert "Town02" not in line and "Town03" not in line


def test_the_town_rotates_on_a_reset_not_mid_episode():
    s = src()
    i_rot = s.index("town_i = (town_i + 1) % len(TOWNS)")
    i_reset = s.index("obs, info = env.reset()\n            vid.start(episodes)")
    assert i_rot < i_reset, "the town must change before the reset that uses it"


def test_exactly_one_encoder_is_constructed():
    """One frozen DEBiT, used by the env. Two would be wasted memory and an
    invitation to let them diverge."""
    s = src()
    assert s.count("DebitTinyEncoder(") == 1
