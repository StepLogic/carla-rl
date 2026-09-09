"""Episode-level Hindsight Experience Replay for the CARLA point-goal task.

Ported from the peer Habitat repo's episode_her.PointGoalRelabelBuffer, keeping
its contract and both of the bugs it paid for. Only the point-goal path is
carried over: their other three buffer classes stage raw panoramas and run a
batched DINOv2 pass at flush, which is an I/O optimisation for 602 KB/step
images and has no bearing on a 12-d goal block.

THE CONTRACT. A CARLA navigation goal is a pure function of (vehicle pose, goal
position), so a hindsight goal is recomputed ARITHMETICALLY -- no encoder, no
staged frames, no simulator access. Critically, the recomputation calls the
wrapper's OWN transform (VAEGoalWrapper.goal_block) rather than a second copy,
so relabelled states cannot drift from on-policy ones.

The VAE latent is NOT recomputed and does not need to be: it encodes what the
camera SAW, which is unchanged by which goal we later credit the step to. Only
the "goal" key is rewritten; "latent" rides along inside the stored transition.

TWO BUGS INHERITED AS GUARDS -- both were measured in the source repo:

  (a) FREE POSITIVES. If hindsight goals may be drawn closer than the radius
      that pays the goal reward, every such relabel is a full-reward positive
      requiring no travel. They measured 827 of 1294 relabels (63.9%) at a 0.2 m
      floor under a 0.8 m payout radius, and the critic's Q went +11.7 -> -3.95
      while success halved. min_disp is therefore TIED to goal_radius here and
      asserted, not merely documented.

  (b) A DICT IS TRUTHY. Their `hit` read info["collision"], a non-empty dict, so
      it was True on every step; relabelled transitions got masks=0/dones=1 and
      the critic could never bootstrap through them. This module asserts `hit`
      is a genuine bool. CARLA's config computes
      `self.collision = 'collision' in sensor_data.keys()`, which already is
      one -- the assert keeps it that way.
"""
import numpy as np


class EpisodeRelabelBuffer:
    """Wraps a replay buffer, inserting hindsight copies at episode end.

    Episode-level rather than sample-level: the hindsight goal is fixed AT
    INSERT rather than redrawn on every sample. That costs goal diversity, and
    n_relabel buys it back as extra rows. The peer repo chose this to amortise
    an encoder pass; here the motive is simpler -- relabelling needs the whole
    episode's positions to pick a reachable future goal, which only exist once
    the episode ends.
    """

    def __init__(self, buffer, state_fn, goal_radius, k_goal, k_collision,
                 n_relabel=1, min_disp=None, max_disp=np.inf, seed=0,
                 max_steps=10000, k_lane=None, k_dist=0.0,
                 k_dist_target_speed=0.0, k_dist_gate_mode="positive",
                 k_speed=0.0, speed_limit_kmh=0.0,
                 k_dist_delta_mode="positive", k_living=0.0,
                 time_to_goal=False, step_cost=0.1):
        # TIME-TO-GOAL MODE. Every coefficient above belongs to SparseGoalReward
        # and is DEAD when this is on -- the relabelled reward becomes
        # `0 on arrival, -step_cost otherwise`, matching TimeToGoalReward.
        #
        # WITHOUT THIS FLAG, ENABLING HER UNDER TIME_TO_GOAL IS A CORRECTNESS
        # BUG, not a tuning choice. The on-policy rows in the buffer carry
        # 0-on-arrival / -0.1-per-step and never terminate on a collision; the
        # relabelled rows would carry +600-on-arrival / 0-per-step and mask=0 on
        # collision. That is precisely the "two inconsistent reward functions
        # over one set of states" failure this module's own comments warn about
        # three separate times, at a 600-vs-0 magnitude on the same event.
        #
        # The earlier claim in run_timetogoal.sh -- that a time-to-goal return
        # "would need the whole remaining trajectory recosted, which HER cannot
        # do" -- was wrong, and that is why HER was left off. The reward is
        # LOCAL: -step_cost on every step, 0 on the step that arrives. It
        # depends only on whether pos[i+1] is inside goal_radius of the
        # hindsight goal, which is the same test the sparse path already makes.
        # Nothing downstream of step i is consulted, so there is nothing to
        # recost.
        self.time_to_goal = bool(time_to_goal)
        self.step_cost = float(step_cost)
        self.buffer = buffer
        # The wrapper's OWN goal-block transform. See module docstring.
        self.state_fn = state_fn
        self.goal_radius = float(goal_radius)
        self.k_goal = float(k_goal)
        self.k_collision = float(k_collision)
        # Lane departure is goal-INDEPENDENT, exactly like a collision: the
        # vehicle left its lane regardless of which goal the step is later
        # credited to. Omitting it here would make relabelled rows describe a
        # different reward function than the on-policy ones, and the critic
        # would be fitting two inconsistent MDPs over the same states.
        self.k_lane = float(k_collision if k_lane is None else k_lane)
        # PROGRESS SHAPING MUST BE RECOMPUTED HERE TOO.
        #
        # The env pays k_dist * (d_prev - d_now) toward the CURRENT goal. A
        # hindsight transition has a DIFFERENT goal, so its progress term is a
        # different number -- carrying the on-policy one over, or omitting it,
        # both leave relabelled rows describing a reward function the env never
        # implements, and the critic then fits two inconsistent MDPs over the
        # same states. Euclidean delta is recomputable from stored positions
        # alone, which is the whole reason this term was chosen over the env's
        # road-projected one.
        self.k_dist = float(k_dist)
        # The speed gate must be recomputed here too. Speed is a property of
        # the STEP, not of the goal, so it is stored per step and reused
        # unchanged for every hindsight goal drawn from that step -- but the
        # PROGRESS it gates is goal-dependent, so the gate has to be reapplied
        # rather than the on-policy result carried over.
        self.k_dist_target_speed = float(k_dist_target_speed)
        self.k_dist_gate_mode = str(k_dist_gate_mode)
        # Speeding is GOAL-INDEPENDENT -- the vehicle was over the limit
        # whatever goal the step is later credited to -- so unlike the progress
        # term this needs no per-goal recomputation, only the same coefficients.
        # Omitting it here would still leave relabelled rows describing a
        # different reward function than the env's.
        self.k_speed = float(k_speed)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.k_dist_delta_mode = str(k_dist_delta_mode)
        # The per-step time penalty is GOAL-INDEPENDENT and charged on every
        # step regardless of which goal it is credited to, so a relabelled row
        # must carry it too. Omitting it would make hindsight rows cheaper per
        # step than on-policy ones and the critic would fit two reward
        # functions over the same states.
        self.k_living = float(k_living)
        self.n_relabel = int(n_relabel)
        self.max_disp = float(max_disp)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        # GUARD (a). Defaults to the payout radius and may never sit below it.
        self.min_disp = float(goal_radius if min_disp is None else min_disp)
        if self.min_disp < self.goal_radius:
            raise SystemExit(
                f"HER min_disp={self.min_disp} is below goal_radius="
                f"{self.goal_radius}. Every hindsight goal drawn in "
                f"[{self.min_disp}, {self.goal_radius}) would pay the full goal "
                "reward for no travel -- measured at 63.9% of all relabels in "
                "the source repo, and it halved success. Raise min_disp.")

        self._reset_episode()
        self.last_pairs = []
        self.n_relabelled = 0
        self.n_free_positives = 0
        # How many hindsight rows actually pay k_goal. This is THE number for a
        # sparse-reward run: if relabels almost never pay, the buffer contains
        # essentially no goal-reaching signal and no amount of reward tuning
        # will teach navigation. Success rate alone cannot distinguish "the
        # agent ignores the goal" from "the goal is not in the data".
        self.n_paying = 0
        self.n_onpolicy_success = 0
        self.n_legs = 0

    def _reset_episode(self):
        self._txns, self._fields, self._pos, self._hit = [], [], [], []
        self._lane = []
        self._speed, self._target = [], []
        # Index just past the end of each LEG. A leg is the stretch between two
        # goal arrivals; with chaining off there is exactly one and everything
        # below is a no-op.
        self._leg_ends = []

    def end_leg(self):
        """Close the current leg. Called by the trainer on every arrival.

        WHY RELABELLING MUST NOT CROSS A LEG BOUNDARY -- the trap that most
        likely killed the peer repo's HER8 run.

        Without this the whole chained episode is ONE relabelling unit, and
        _pick_goal's candidate filter is purely a EUCLIDEAN displacement band
        with no temporal bound. In a long chain the vehicle routinely drives
        back near where it has already been, so a transition at step 10 can be
        relabelled onto a state 250 steps later that merely happens to be
        spatially close -- and the arrival test then pays the full k_goal for a
        "goal" that actually took 240 steps to reach. That is false-positive
        supervision, and short single-goal episodes were the only thing
        suppressing it.

        It is also a correctness issue independent of distance: after an
        arrival the goal CHANGES, so a state's stored goal block refers to a
        different goal than any hindsight target drawn from a later leg.
        """
        n = len(self._txns)
        if n and (not self._leg_ends or self._leg_ends[-1] != n):
            self._leg_ends.append(n)

    def __len__(self):
        return len(self.buffer)

    # -- per step ----------------------------------------------------------
    def record(self, transition, fields, pos, hit, lane=False,
               speed_kmh=0.0, target_speed_kmh=0.0):
        """Stage one step. `fields` is what state_fn needs, ~100 bytes."""
        if not isinstance(hit, (bool, np.bool_)):
            # GUARD (b).
            raise TypeError(
                f"hit must be a bool, got {type(hit).__name__}={hit!r}. A dict "
                "or array is truthy on every step, which would mark every "
                "relabelled transition terminal and stop the critic "
                "bootstrapping through them.")
        if len(self._txns) >= self.max_steps:
            for lst in (self._txns, self._fields, self._pos, self._hit,
                        self._lane, self._speed, self._target):
                lst.pop(0)
        self._txns.append(transition)
        self._fields.append(dict(fields))
        self._pos.append(np.asarray(pos, np.float64)[:2])
        self._hit.append(bool(hit))
        self._lane.append(bool(lane))
        self._speed.append(float(speed_kmh))
        self._target.append(float(target_speed_kmh))
        # The on-policy transition goes in immediately; only the hindsight
        # copies wait for the episode to end.
        self.buffer.insert(self._clean(transition))

    # -- at episode end ----------------------------------------------------
    def _pick_goal(self, i, n, pos):
        """A future step whose displacement from step i falls in the band.

        Uniform over the FUTURE would concentrate hindsight goals one or two
        steps ahead -- already inside the arrival radius -- so the reward fires
        immediately and the policy learns "you are already there, do nothing".
        Sampling by displacement is what prevents that.
        """
        pos_i = pos[i]
        cand = [t for t in range(i + 1, n)
                if self.min_disp <= float(np.linalg.norm(pos[t] - pos_i)) <= self.max_disp]
        if not cand:
            return None
        return int(self.rng.choice(cand))

    def flush_episode(self):
        """Insert hindsight copies for the finished episode, then clear."""
        n = len(self._txns)
        if n < 2:
            self._reset_episode()
            return 0

        # Legs partition the episode. The final leg is closed here so a chain
        # that ended on a collision rather than an arrival is still bounded.
        bounds = list(self._leg_ends)
        if not bounds or bounds[-1] != n:
            bounds.append(n)

        pairs = []
        for _ in range(self.n_relabel):
            start = 0
            for end in bounds:
                # `end` is the exclusive bound passed to _pick_goal, so a
                # hindsight goal can never be drawn from a later leg.
                for i in range(start, max(start, end - 1)):
                    tp = self._pick_goal(i, end, self._pos)
                    if tp is not None:
                        pairs.append((i, tp))
                start = end

        inserted = 0
        for (i, tp) in pairs:
            goal = self._pos[tp]
            row = self._relabelled(i, goal, n)
            self.buffer.insert(row)
            inserted += 1

        self.last_pairs = pairs
        self.n_legs += len(bounds)
        self.n_relabelled += inserted
        self._reset_episode()
        return inserted

    def _relabelled(self, i, goal, n):
        """One hindsight transition: rewrite ONLY the goal key and the reward."""
        row = self._clean(self._txns[i])

        obs_goal = self.state_fn({**self._fields[i], "goal_pos": goal})
        nxt_goal = self.state_fn({**self._fields[min(i + 1, n - 1)],
                                  "goal_pos": goal})

        # Rewrite exactly one key of each observation. "latent" is what the
        # camera saw and is independent of which goal we credit the step to.
        row["observations"] = dict(row["observations"])
        row["next_observations"] = dict(row["next_observations"])
        row["observations"]["goal"] = obs_goal
        row["next_observations"]["goal"] = nxt_goal

        # SPARSE reward, recomputed with the SAME coefficients the env uses.
        # Different coefficients here would make the critic fit two different
        # reward functions over the same states.
        #
        # PAYOUT TESTS THE NEXT STATE, NOT THE CURRENT ONE. For a transition
        # (s_i, a_i, s_i+1) the reward describes arrival at s_i+1. The source
        # repo tested pos[i], which collides with _pick_goal's band: that filter
        # guarantees ||pos[tp] - pos[i]|| >= min_disp while the payout required
        # the same quantity to be < goal_radius, and min_disp == goal_radius --
        # complementary comparators on one quantity, so `reached` was
        # unreachable by construction and EVERY relabel paid 0. Confirmed
        # empirically by that repo's author: 28 relabels, 0 paying; 6 paying
        # once the test moved to pos[i+1].
        #
        # Testing pos[i+1] also decouples the two constraints -- the band
        # constrains pos[i], the payout tests pos[i+1] -- which is what lets
        # min_disp >= goal_radius hold without being self-defeating.
        #
        # ANY DENSE SHAPING TERM ADDED HERE MUST USE THE SAME TIMESTEP. Their
        # k_dist term reads pos[i-1]/pos[i]; mixing that with a pos[i+1] payout
        # makes the sparse and dense terms describe different steps. This buffer
        # is deliberately sparse-only so the question cannot arise.
        d_next = float(np.linalg.norm(self._pos[min(i + 1, n - 1)] - goal))
        reached = d_next < self.goal_radius
        hit = self._hit[i]
        lane = self._lane[i]

        if self.time_to_goal:
            # THE WHOLE REWARD. Mirrors TimeToGoalReward.step exactly.
            #
            # `hit` and `lane` are read above and deliberately NOT used: under
            # this reward a collision costs only the time it loses, so it
            # neither pays a penalty nor ends the episode, and a relabelled row
            # that set masks=0 there would stop the critic bootstrapping through
            # states the on-policy rows bootstrap through freely.
            #
            # Truncation has no hindsight analogue: the step limit applies to
            # the REAL episode, and a hindsight goal drawn from inside it is
            # always reached within the trajectory that was actually flown. So
            # the only terminal event here is arrival, and mask=1 elsewhere is
            # correct rather than an approximation.
            row["rewards"] = np.float32(0.0 if reached else -self.step_cost)
            row["masks"] = np.float32(0.0 if reached else 1.0)
            row["dones"] = np.float32(1.0 if reached else 0.0)
            if reached:
                self.n_paying += 1
            if reached and float(
                    np.linalg.norm(self._pos[i] - goal)) < self.goal_radius:
                self.n_free_positives += 1
            return row

        # SAME TIMESTEP AS THE PAYOUT: the arrival test reads pos[i+1], so the
        # progress term reads ||pos[i] - g|| - ||pos[i+1] - g||. Using
        # pos[i-1]/pos[i] here -- as the source repo does -- would make the
        # sparse and dense halves of one reward describe different steps.
        progress = 0.0
        if self.k_dist:
            d_cur = float(np.linalg.norm(self._pos[i] - goal))
            delta = d_cur - d_next
            # SAME DELTA_MODE AS THE ENV. If the env clips at 0 and this does
            # not, relabelled rows describe a reward function the env never
            # implements and the critic fits two inconsistent MDPs over one set
            # of states.
            if self.k_dist_delta_mode == "signed":
                progress = self.k_dist * delta
            elif self.k_dist_delta_mode == "abs":
                progress = self.k_dist * abs(delta)
            else:
                progress = self.k_dist * max(delta, 0.0)
            target = self.k_dist_target_speed or (
                self._target[i] if i < len(self._target) else 0.0)
            if target > 0.0 and i < len(self._speed) and self._speed[i] >= target:
                progress = (0.0 if self.k_dist_gate_mode == "both"
                            else min(progress, 0.0))

        speed_pen = 0.0
        if self.k_speed and i < len(self._speed):
            limit = self.speed_limit_kmh or (
                self._target[i] if i < len(self._target) else 0.0)
            if limit > 0:
                speed_pen = self.k_speed * max(0.0, self._speed[i] - limit)

        reward = ((self.k_goal if reached else 0.0)
                  - (self.k_collision if hit else 0.0)
                  - (self.k_lane if lane else 0.0)
                  + progress - speed_pen - self.k_living)
        done = bool(reached or hit or lane)

        row["rewards"] = np.float32(reward)
        row["masks"] = np.float32(0.0 if done else 1.0)
        row["dones"] = np.float32(1.0 if done else 0.0)

        if reached:
            self.n_paying += 1
        if reached and float(np.linalg.norm(self._pos[i] - goal)) < self.goal_radius:
            # Started inside the payout radius: a free positive. Guard (a) makes
            # this unreachable, so counting it is a tripwire, not a fallback.
            self.n_free_positives += 1
        return row

    @staticmethod
    def _clean(t):
        return {k: v for k, v in t.items() if not k.startswith("_")}

    def stats(self):
        frac = (self.n_free_positives / max(1, self.n_relabelled))
        return {
            "her_legs": self.n_legs,
            "her_relabelled": self.n_relabelled,
            "her_free_positive_frac": frac,
            "her_paying": self.n_paying,
            "her_paying_frac": self.n_paying / max(1, self.n_relabelled),
        }
