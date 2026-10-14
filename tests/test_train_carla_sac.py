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
def test_the_env_is_built_WITHOUT_an_encoder():
    """The learner's act() encodes the same pair a moment later through the same
    adapters. Letting the wrapper encode too doubles the per-step cost for a
    vector the loop never reads."""
    s = src()
    assert re.search(r"build_env\([^)]*encoder=None", s, re.S), \
        "build_env must be called with encoder=None"


def test_one_encoder_instance_is_shared_with_the_learner():
    """Two instances would mean two sets of adapters: the rollout would act with
    parameters the update never touches."""
    s = src()
    assert "debit_encoder=enc" in s
    assert s.count("DebitTinyEncoder(") == 1


def test_the_adapters_are_added_before_the_learner_is_built():
    """build_visual wraps whatever the encoder's blocks are at construction. Add
    the adapters afterwards and the learner's optimiser never sees them."""
    s = src()
    assert s.index("add_encoder_adapters") < s.index("GatedRecurrentSAC(cfg)")


# ------------------------------------------------------------- the row --
def test_the_mode_bit_is_appended_to_the_state_block():
    s = src()
    assert 'np.asarray(o["mode"]' in s and "concatenate" in s
    assert 'np.asarray(o["vector"]' in s, \
        'the state block is the 4-d proprioceptive vector, not the env\'s 5400-d features'
    assert 'o["state"]' not in s


def test_mode_index_points_at_the_LAST_state_column():
    """The bit is appended, so the index must be the last column. Off by one and
    the learner masks on a velocity."""
    s = src()
    assert "mode_index=state_dim - 1" in s
    assert 'state_dim = int(np.shape(obs["vector"])[0]) + 1' in s


def test_the_single_critics_view_is_the_MODE_BIT():
    """User 2026-10-13: one critic for both reward modes, and the mode bit
    observed by it. Without that column the critic predicts two different
    returns from otherwise identical inputs and learns their average -- which
    does not fail, it just reads as mediocre. There is no other goal column:
    with goal_tokens the goal is an image the GRU has already folded in."""
    s = src()
    assert "one_critic=os.environ.get(\"ONE_CRITIC\", \"1\") == \"1\"" in s
    assert "goal_cols=(state_dim - 1,), explore_cols=()" in s
    assert "mode_index=state_dim - 1" in s          # the same column


def test_the_row_carries_the_action_that_was_ACTUALLY_taken():
    """prev_a must be updated from `action`, not from the policy's next output
    or a zero: the row's previous-action slot is part of the observation."""
    s = src()
    assert "obs, info, row, prev_a = nobs, ninfo, nrow, action" in s


def test_the_previous_action_is_zeroed_at_every_reset():
    s = src()
    assert s.count("prev_a = np.zeros(2, np.float32)") >= 2, \
        "zeroed at start AND after each reset"


# ------------------------------------------------------------- the goal --
def test_the_goal_frame_is_latched_per_EPISODE():
    """Once at start and once per reset -- never per step, which would make the
    target drift under the policy."""
    s = src()
    assert s.count('pxr.goal_block(info["debit_goal"])') == 2
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
    for want in ("town=", "seed=", "action box", "checkpoints ->", "reward:"):
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
    i_end = s.index('end = str(info.get("goal_end"')
    assert s.index("vid.save(end)") > i_end


def test_recording_happens_before_the_observation_is_advanced():
    """info must still be the step that was just acted on. Record after `obs,
    info = nobs, ninfo` and every frame is one step late -- the stale-frame bug
    that cost an earlier run its perception."""
    s = src()
    assert s.index('vid.add(info["debit_cur"]') < s.index("obs, info, row, prev_a = nobs")
