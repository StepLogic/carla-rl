#!/usr/bin/env bash
# Minimum-time-to-goal task.
#
#   r = -0.1                          every step
#   r =  0        and terminate       on arrival
#   r = -0.1 * (remaining + 1)  and terminate   on collision OR lane departure
#   episode ends anyway at 1500 steps
#
# Every terminal failure costs exactly 0.1 * 1500 = 150, identical to running
# out of time, so there is never an incentive to trigger one. Return lies in
# [-150, 0] and every unit of it is time.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla|[r]un_|[w]atch_policy" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- the task ----------------------------------------------------------------
#   r = -0.1              every step
#   r =  0   + terminate  on arrival
#            + truncate   at 1000 steps
#
# Collision and lane departure do NOT end the episode and carry no penalty.
# Their cost is the time they lose: a car stuck against a wall stops making
# progress and pays the full -100; one that clips a kerb and drives on pays
# only the seconds it lost. Return lies in [-100, 0] and every unit is time.
export TIME_TO_GOAL=1
# ONE UNIT OF COST PER STEP. r = -1.0 every step, 0 on arrival, so the return
# is the step count itself in negative units and V(s) reads directly as
# -(steps to goal). At 0.1 the same quantity was in tenths and every readout
# had to be divided.
#
# THIS IS A 10x RESCALE OF EVERY RETURN IN THIS ARM, and two things are pinned
# to that scale -- see TQC_KAPPA below, and vae_nav.value_distance, whose
# STEP_COST_DEFAULT is still 0.1. Any graph built from a critic trained here
# MUST be passed step_cost=1.0 explicitly, or every edge weight comes out 10x
# too long. run_sorb_agent.sh keeps its own STEP_COST=0.1 and is unaffected.
export STEP_COST=1.0
# EPISODE BUDGET SCALED TO THE BAND. MEASURED over 75 recorded eval episodes:
# 0.283 m per step (median, max 0.327) at ~11 km/h on a 0.1 s timestep. The
# 50 m ceiling therefore needs >= 177 steps of PERFECT driving; 500 steps is
# 141 m of travel, ~2.8x the longest goal -- the same headroom ratio the 200 m
# band had at 2000 steps.
#
# RAISED WITH THE BAND, 250 -> 500. Left at 250 a 50 m goal would have had only
# 1.4x headroom, which this script's own history records as the setting where
# "most episodes would have timed out for reasons unrelated to the policy".
#
# THE HEADROOM RATIO IS THE POINT, not the absolute number. At 2000 steps
# against a 25 m band a successful episode cost ~88 steps while a wedge on step
# 5 billed the remaining ~1995 -- a 22:1 gap between failure and success for
# TQC's quantiles to span, where the original 200 m design had 2.8:1. Keeping
# the cap at ~2.8x the perfect-driving time of the LONGEST goal keeps that gap
# constant as the band moves.
#
# Return range is [-STEP_COST * 500, 0] = [-500, 0] at STEP_COST=1.0. The wedge
# bill stays exact because it is computed from max_steps, not hardcoded.
export TTG_MAX_STEPS=750
# Must MATCH TTG_MAX_STEPS: the outer TimeLimit wrapper is the only other
# source of truncation, and a shorter one would cut episodes before the
# wrapper's own limit, making the reported return shorter than the task allows.
export MAX_EPISODE_STEPS=750

# The sparse reward's terms are all DEAD under TIME_TO_GOAL -- the wrapper
# replaces the reward wholesale. Pinned to 0 so nothing is silently carried.
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0

# ---- actions: ABSOLUTE ---------------------------------------------------------
# The action IS the control: steer and throttle are set directly each step.
#
# This gives up the structural smoothness the incremental space provided
# (MEASURED under random actions: p95 |accel| 0.15 m/s^2 incremental vs 196.0
# absolute on the old trained policy, 0.8% vs 79.5% harsh steps). Nothing in
# the time-to-goal reward penalises action magnitude or rate, so bang-bang
# control is once again permitted -- and rewarded, since less time is the whole
# objective. If the driving turns twitchy again, that is why.
#
# The magnitude clips are KEPT. They bound speed and steering angle even though
# they no longer bound the RATE of change: at throttle 0.6 this vehicle settles
# around 22 km/h.
export INCREMENTAL_ACTIONS=0
# THROTTLE CAPPED TO 0.4. This is the ONLY speed limit in the arm: done_speed
# is hardcoded False ("speed-based termination (currently disabled)"), and the
# minimum-time objective actively rewards going faster, so nothing else stops
# the car. MEASURED at the old cap: throttle 0.6 settles this vehicle at a
# 22.4 km/h median.
#
# THE THROTTLE -> SPEED CURVE BELOW 0.6 IS NOT MEASURED. The step budget below
# is sized on an ASSUMED 0.7x speed factor; if episodes start timing out
# systematically rather than failing on off_road or collision, that assumption
# was wrong and the budget, not the cap, is what needs revisiting.
export MAX_STEER=0.5 MAX_THROTTLE=0.4
# PROPRIOCEPTIVE STATE MATCHES HABITAT: the 12-d goal block only, no extra
# vector. Theirs is [fwd, left, sin(yaw), cos(yaw), a_{t-3..t}] and ours is
# byte-for-byte the same shape. The 4-vector was added only because the
# per-step reward was speed-gated and the gate would have been unobservable;
# under TIME_TO_GOAL there is no speed gate (K_DIST=K_SPEED=0), so it goes.
#
# Velocity is inferred from the THROTTLE HISTORY inside the goal block, which
# now records the applied control rather than the action delta.
export OBS_WITH_VECTOR=0

# ---- KAPPA MUST TRACK THE REWARD SCALE ---------------------------------------
# tqc_kappa() derives 60 from K_GOAL=600, which is meaningless here. Returns lie
# in [-500, 0] and per-step rewards are 1.0. quantiles.py:72 computes
# `huber / kappa`, so too LARGE a kappa divides the critic gradient away and the
# critic barely learns -- MEASURED on the lane-follow run, where critic_loss
# printed as 0.00 at 22k steps at kappa=60.
#
# 10.0, RAISED FROM 1.0 WITH STEP_COST 0.1 -> 1.0. Kappa is the Huber knee: the
# error magnitude below which the loss is quadratic and above which it goes
# linear. Rescaling every reward by 10x multiplies every TD error by 10, so at
# kappa=1 essentially the whole error distribution would sit in the LINEAR
# region and the critic would train under L1 everywhere -- gradient magnitude
# saturated, no more signal from a large error than a small one. Scaling kappa
# with the reward keeps the same quadratic/linear split the arm had at 0.1.
export TQC_KAPPA=10.0

# ---- RETURN BOUNDS, now that gamma=1 has more truncation to survive ---------
# Every return lies in [-STEP_COST * TTG_MAX_STEPS, 0] BY CONSTRUCTION, so a
# target outside it is not unlikely but UNREALISABLE, and clipping adds
# information rather than bias (critic_updater.py:12-30).
#
# WHY NOW. gamma=1 has no contraction: V(s) = -c + V(s') has no bounded fixed
# point on a state the agent never escapes, and TRUNCATED episodes bootstrap
# (masks=1). MEASURED on the earlier time-to-goal run: as the non-arriving
# fraction went 0.10 -> 0.50, q drifted -20.6 -> -33.8, TD error quadrupled
# 0.082 -> 0.318 and success fell 0.700 -> 0.500. The stall makes this worse,
# because crossings no longer end episodes and so more of them time out.
# Collision-terminal pushes the other way -- every masks=0 terminal removes an
# unbounded chain -- but the clip is the cheap direct guard.
#
# THE UPPER BOUND STAYS 0 AND THAT IS ONLY SAFE WHILE POTENTIAL_K IS SMALL.
# With shaping, V(s) = -c*E[steps] + POTENTIAL_K * remaining(s), and at ~0.283
# m/step a metre of route costs ~3.5 steps. So V <= 0 requires
# POTENTIAL_K < ~3.5; above that the agent is paid net-positive for driving,
# time pressure inverts, and this clip starts biasing the critic.
export TQC_TARGET_MIN=-750 TQC_TARGET_MAX=0

# UNDISCOUNTED. gamma=1 makes Q exactly the cost-to-go: Q(s) = -0.1 * steps
# remaining to the goal, readable directly as negative time. At gamma=0.99 the
# horizon is 100 steps, Q saturates at -10, and Q(400 steps) vs Q(1000 steps)
# differ by 0.18 -- unusable over the range this curriculum reaches.
#
# Safe ONLY because every episode ends within TTG_MAX_STEPS, so the
# undiscounted return is bounded by construction at -100.
export DISCOUNT=1.0

# ---- curriculum: 5 -> 113 m ----------------------------------------------------
# TRAIN Town02, EVALUATE Town01. Town01 is a TRAINING-POOL town (the frozen
# split holds out Town03 and Town10HD), but this run never trains on it, so
# the eval is a genuine cross-town test of the policy -- NOT the held-out
# claim, which needs Town03/Town10HD.
#
# MIND THE LABEL: run_evaluation tags the line "IN-DIST" whenever EVAL_TOWNS is
# set, because that override exists for the overfitting test. Here it is
# cross-town, so read the [Eval] line's town breakdown, not the tag.
export ENV_TOWNS=Town02 EVAL_TOWNS=Town01
# MINIMUM GOAL DISTANCE 20 m. Below this the task is a nudge rather than a
# navigation problem: at ~22 km/h a 5 m goal is ~8 steps, so most of the band
# was being spent on episodes the policy could solve without steering. The
# minimum is a FLOOR, not a stage -- the curriculum only moves GOAL_DIST_MAX.
#
# This resets the eval band, so eval numbers from here are NOT comparable to
# the 0.883/0.833/0.800 series before it: those sampled from 5 m up.
# POINT-GOAL, 100-200 m. The goal is a RELATIVE POSITION -- the 12-d block is
# [fwd, left, sin(yaw), cos(yaw), 4x(steer, throttle)] -- so this is point-goal
# navigation, not image-goal, despite the config filename.
#
# THE BAND IS ALONG-ROAD DISTANCE, not euclidean: carla_core walks forward
# through waypoints by `lenghts` metres, so a 200 m goal is 200 m of ROAD.
#
# 20-25 m, PINNED: START == TARGET so the curriculum never widens the band.
# The floor and the ceiling are 5 m apart, so every episode is the same task.
#
# WHY THIS BAND. At the nominal ~22 km/h (0.283 m/step measured, 0.1 s tick) a
# 22 m goal is ~80 steps -- long enough to need steering through a junction,
# short enough that the wedge's ABSORBING failure has little road to compound
# over. MEASURED on 375 eval episodes (results/policy_eval.json): success was
# 0.945 at 0-25 m and fell monotonically to 0.550 at 100-150 m, with EVERY
# failure a lane crossing. Distance was the variable that mattered.
#
# This replaces GOAL_DIST_MIN=100/START=100/TARGET=200, so eval numbers from
# here are NOT comparable to the 0.417-0.567 series measured at 20-113 m.
export GOAL_DIST_MIN=20.0 GOAL_DIST_START=50.0 GOAL_DIST_TARGET=50.0
export GOAL_DIST_MUL=1.25 CURRICULUM_SUCCESS=0.5 CURRICULUM_MIN_EPISODES=50
export CURRICULUM_DEMOTE=0.15 CURRICULUM_STALL_EPISODES=400
# GOALS AT JUNCTIONS. The default sampler walks forward from a uniformly random
# waypoint, so in Town02 most goals sit on straight road and the agent is rarely
# asked to cross a junction -- which is where the turns, the legal lane
# crossings and the wrong_dir termination all live. 0.7 places the origin one
# goal-distance BEFORE a junction on 70% of episodes, so the route through it,
# leaving 30% straight-road.
#
# RAISED FROM 0.6 TOGETHER WITH THE 20-25 m BAND, and the two belong together:
# a 22 m goal placed before a junction is almost entirely the turn, whereas at
# 100 m the junction was a small fraction of a long straight drive. The
# junction is now the task rather than an incident within it.
#
# THIS MAKES THE LANE_STRICT=0 JUNCTION EXEMPTION LOAD-BEARING. A Broken line
# crossed inside a junction is the turn itself; at 70% junction episodes,
# making that fatal would leave most of the run unwinnable by any driver.
export JUNCTION_GOAL_FRAC=0.7
export SHORT_HORIZON=1
export MAX_DISTANCE_M=1200        # must exceed the longest goal, or the env's
                                  # own distance terminator ends the episode first

# ---- environment ---------------------------------------------------------------
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
# WEDGE ON LANE CROSSING. Crossing a marking (or leaving the road) freezes the
# vehicle for the rest of the episode -- compute_action returns full brake
# whatever the policy asks. The cost is paid purely in TIME, which keeps the
# critic's value an honest cost-to-go for the SORB graph; a reward penalty
# would have made V(s) mean "steps to goal PLUS a fine" instead.
export LANE_WEDGE=1
# ONLY JUNCTION-INTERIOR CROSSINGS ARE FORGIVEN. A Broken line crossed inside a
# junction is the turn; the same line crossed on open road is the vehicle
# drifting out of lane, and stays fatal.
#
# THIS CURRENTLY FORGIVES NOTHING, AND THE MEASUREMENT SAYS WHY.
#
# The theory was that the wedge fired on legal junction turns: JUNCTION_GOAL_FRAC
# =0.6 routes 60% of episodes through a junction, and at LANE_STRICT=1 the
# marking breakdown was Broken=0.54, NONE=0.20, Solid/Curb/Grass=0.00 -- no
# genuine violations, so the crossings looked like legal turns.
#
# Tagging each crossing with WHERE it happened refuted that outright:
#
#   [Why 30] lane_any=0.77  mark:Broken@road=0.50  mark:NONE@road=0.27
#
# Every crossing @road. Not one @junc, over 30 episodes. The agent is not being
# punished for turning through junctions -- it is drifting out of lane on
# straight road, which is a driving-competence problem and not an
# environment-rules one. The exemption is kept because it is correct for when
# junction crossings DO occur, but it is not what is capping success.
export LANE_STRICT=0
# OFF-ROAD IS FATAL. Leaving the drivable surface freezes the vehicle.
#
# Briefly disabled on the theory that get_waypoint(project_to_road=False) was
# firing spuriously inside junctions. The junction-tagged measurement killed
# that theory -- see LANE_STRICT above, every crossing was tagged @road -- and
# the reason to keep it is unchanged: under a minimum-time objective a shortcut
# across grass is REWARDED whenever it is shorter, and off_road is the only
# signal that catches leaving the road where nothing is painted. The lane
# sensor fires on crossing a PAINTED marking, so a kerb-hop onto a median never
# triggers it at all.
export WEDGE_ON_OFF_ROAD=1
# A WEDGED EPISODE IS ENDED IMMEDIATELY AND BILLED FOR THE TIME IT WOULD HAVE
# BURNED. The vehicle is frozen under full brake and the flag is never cleared,
# so it cannot arrive and will certainly run to TTG_MAX_STEPS -- simulating
# those ~950 identical frozen frames costs wall-clock and teaches nothing.
#
# The bill is EXACT, so the episode return is unchanged at -STEP_COST *
# TTG_MAX_STEPS: this buys simulation time and nothing else. Ending early is
# worth precisely zero, which is what stops it becoming an escape.
# tests/test_wedge_truncate.py asserts billing == simulating at every step the
# wedge can fire; if that ever drifts, the wedge silently becomes an incentive.
export WEDGE_TRUNCATE=1
# HINDSIGHT OFF. Kept correct rather than deleted -- EpisodeRelabelBuffer's
# time_to_goal path stays wired and tested, so EPISODE_HER=1 remains a valid
# switch; it just is not what this run uses.
#
# MEASURED over the completed 500k run: 210,751 relabelled rows at a 3.6%
# paying fraction, i.e. 96% of hindsight rows were non-terminal masks=1
# bootstraps under gamma=1. Success plateaued at 0.44 and the lane-crossing
# rate never moved off 0.58 across 1600 episodes, so those rows bought nothing
# measurable while feeding the one structure most likely to destabilise an
# undiscounted critic.
#
# The comment that stood here -- that a time-to-goal return
# "would need the whole remaining trajectory recosted, which HER cannot do" --
# was WRONG, and it is why this was off. The reward is LOCAL: -0.1 on every
# step, 0 on the step that arrives, terminal only on arrival. A relabelled row
# needs one test, whether pos[i+1] lies inside the goal radius of the hindsight
# goal, and nothing downstream of step i is consulted.
#
# EpisodeRelabelBuffer takes time_to_goal=1 from TIME_TO_GOAL to select that
# path. Without it, hindsight rows would carry the SPARSE reward (+600 arrival,
# -5 collision, mask=0 on collision) into a buffer whose on-policy rows pay 0
# on arrival, -0.1 per step, and never terminate on collision.
export EPISODE_HER=0
# Left at the default (= GOAL_THRESHOLD, 2 m) ON PURPOSE, rather than tracking
# GOAL_DIST_MIN=20. Hindsight goals are the ONLY remaining source of
# short-range value estimates now that env goals start at 20 m, and short-range
# is exactly what the SORB graph consumes: its edge weights are single hops
# between nearby trace nodes, not 20 m routes. Raising this to 20 would leave
# the critic with no data at the distances the graph is built from.
# export HER_MIN_DISP=20.0
export CHAIN_MAX_GOALS=1
# SCRIPTED EXPERT ON 20% OF EPISODES -- the value the peer Habitat trainer uses.
#
# ExpertOracle drives WHOLE episodes with GlobalRoutePlanner + a PID controller,
# emitting (steer, throttle) in the agent's own action space, so its rows are
# ordinary off-policy transitions the buffer cannot distinguish.
#
# WHY IT MATCHES THIS FAILURE. The binding constraint over the whole 500k run
# was lane-keeping on STRAIGHT road -- lane_any pinned at 0.58, flat to two
# decimals across 1600 episodes, split Broken@road 0.39 / NONE@road 0.19 with
# zero @junc. A PID route follower tracks the lane centre by construction, so
# expert episodes put clean lane-keeping trajectories in the buffer, which is
# the one thing 500k steps of self-play never produced.
# RAISED 0.2 -> 0.4. Doubling the share of episodes driven by the scripted
# PID route follower doubles the clean lane-keeping trajectories in the buffer
# -- "the one thing 500k steps of self-play never produced". Cost: more of the
# buffer is states the policy does not itself reach, so the critic trains
# further off-distribution. The /orc vs /pol split in [Why] makes that visible.
export EXPERT_PROB=0.4
# GRADED STALL instead of the absorbing wedge. A lane crossing now freezes the
# car for 50 steps (5 s of a 500-step budget) and hands control back, so drift
# costs time rather than the episode. Off-road still stalls repeatedly and stays
# effectively absorbing -- see STALL_STEPS in the env config for why that
# asymmetry is deliberate. Watch `stalled=` and `stalls/ep=` in [Why]; under
# this setting lane_any will read ~0 because nothing wedges any more.
# LANE INVASION TERMINATES. 0 = the absorbing wedge: a crossing freezes the car
# and the episode ends immediately, billed for every remaining step so it costs
# exactly what running out of time costs.
#
# REVERTED FROM 50 (the graded stall) DELIBERATELY. The stall priced the
# CROSSING EVENT, not the STATE of being out of lane: diff_lane is edge
# triggered, so after its 50 frozen steps the car resumed wherever it stopped --
# possibly in the oncoming lane -- and would not trigger again until it crossed
# ANOTHER line. One crossing bought unlimited wrong-side driving. Nothing else
# caught it either: done_wrong_direction needs WRONG_DIR_TERMINAL=1, fires only
# inside junctions, is skipped in eval, and TimeToGoalReward ignores the env's
# `done` anyway; off_road only fires off the drivable surface entirely.
#
# The stall machinery is kept and tested -- STALL_STEPS>0 re-enables it -- but
# it needs a wrong-lane STATE check before it is sound.
export STALL_STEPS=0

# ---- COLLISION IS TERMINAL, AND BILLED --------------------------------------
# A collision ends the episode and is charged every remaining step, so it costs
# exactly what running out of time costs. The bill is not a penalty -- it is
# what stops the terminal becoming an ESCAPE: with a per-step cost and a
# collision that merely stopped the episode, crashing on step one would return
# -1 against -80 for driving honestly to the goal, and crashing immediately
# would be optimal. tests/test_collision_terminal_and_shaping.py pins the
# equality at every step a crash can happen.
export COLLISION_TERMINAL=1

# ---- POTENTIAL-BASED SHAPING ON ALONG-ROAD PROGRESS -------------------------
# Phi(s) = -POTENTIAL_K * (metres of route left), so the agent is paid per metre
# of ROUTE progress and charged for backtracking. Policy-invariant (Ng et al.
# 1999): it moves credit earlier in time without changing the optimum.
#
# ROAD DISTANCE, NOT EUCLIDEAN, and the difference is measured. On this
# project's CORL Town02 set, trajectory 4's endpoints are 39.8 m apart and
# 621.6 m by road, with trajectories 2, 9 and 3 at ratios 8.8, 8.2 and 5.2. A
# straight-line potential would push the car AWAY from the route in a fifth of
# the episodes -- dense and confidently wrong exactly where the task is hard.
#
# K=1.0. At ~0.283 m/step a perfectly progressing car earns ~0.28 against the
# 1.0 it pays for the step, so progress is worth having and time pressure
# survives. See TQC_TARGET_MAX above for why K must stay well under ~3.5.
#
# THE CRITIC IS NO LONGER RAW NEGATIVE TIME, AND THAT IS ACCEPTED. Decided
# 2026-09-10: feed -V to the SORB graph unmodified rather than subtracting the
# potential. -V stays positive, is a linear rescale of true steps by 0.717 so
# Dijkstra's path ranking is exact, and rises on edges the policy finds hard.
# What must change instead is the CUTOFF: H_STEPS=106 becomes ~76 on this scale.
# See the POTENTIAL_K block in vae_nav/time_to_goal_reward.py.
export ROUTE_POTENTIAL=1
export POTENTIAL_K=1.0
export TARGET_SPEED=20.0

# ---- run -----------------------------------------------------------------------
export POLICY_FOLDER="${POLICY_FOLDER:-carla_timetogoal}"
export MAX_STEPS="${MAX_STEPS:-500000}"
export RESUME="${RESUME:-1}"
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# CarlaCore OWNS THE SERVERS. One for the training env, one for the eval env,
# each on its own random port in 15000-32000, started and stopped by the core
# that uses it. Faster than sharing, and it removes the world reload that a
# shared server would need for a Town01 evaluation on a Town02 run.
#
# CARLA_ROOT is REQUIRED and has no default: carla_core.init_server does a bare
# os.environ["CARLA_ROOT"] and would KeyError on the first env build.
export CARLA_START_SERVER=1
export CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
# pgrep -x CANNOT match here: /proc/PID/comm is capped at 15 chars, so
# 'CarlaUE4-Linux-Shipping' (23) is truncated to 'CarlaUE4-Linux-' and an
# exact-name match never fires. That is why servers survived this trap and
# sat holding ~2.8 GB of GPU after their job had exited. -f matches argv.
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t p < <(pgrep -f '[C]arlaUE4-Linux-Shipping'); [ ${#p[@]} -eq 0 ] && return 0
    for q in "${p[@]}"; do kill -9 "$q" 2>/dev/null; done; sleep 3
  done
}
cleanup() { kill_servers; echo "=== time-to-goal run done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== time-to-goal attempt $ATTEMPT ==="
  kill_servers
  # CARLA_START_SERVER=1 -> CarlaCore owns the servers and this launcher is
  # skipped, or the run would hold two: one here on 2000 and one CarlaCore
  # spawned on a random port. The training env and the eval env then get a
  # server each, which is the point -- eval runs a DIFFERENT town, and sharing
  # one server would mean load_world on every evaluation, the operation known
  # to crash CARLA 0.9.15 (see the ENV_TOWNS note in train_carla_tqc.py).
  if [ "${CARLA_START_SERVER:-0}" != "1" ]; then
    ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
        -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_timetogoal.log ) &
    sleep 40
  else
    echo "[CARLA] CARLA_START_SERVER=1 -- CarlaCore starts its own servers"
  fi
  if python -u train_carla_tqc.py; then
    echo "=== time-to-goal finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
