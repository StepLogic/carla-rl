"""A scripted route-following expert, to seed the sparse-reward buffer.

WHY THIS EXISTS. With the sparse reward (+k_goal inside the arrival radius, 0
otherwise) an agent learns nothing until it reaches a goal at least once. In
Habitat point-nav that happens by accident: the goal is 0.8-2.5 m away and a
random walk finds it. In CARLA it does not. Throttle is bounded, there is no
reverse, goals are tens of metres away along a road network, and a random
policy mostly idles or crashes. The buffer would fill with zero-reward
transitions and the critic would have nothing to fit.

The peer Habitat trainer carries EXPERT_PROB=0.2 -- a navmesh oracle drives 20%
of episodes -- for exactly this reason, even though its task is the easy one.
The CARLA equivalent has to be a route follower, so this wraps the repo's
existing GlobalRoutePlanner (route) and VehiclePIDController (control).

WHOLE EPISODES, NOT INDIVIDUAL STEPS. The expert takes an episode or it does
not. Mixing expert and policy actions inside one episode yields trajectories
that reach the goal only partly under expert control, and under a sparse reward
a partial trajectory that never arrives carries no more signal than a random
one. The point is to put COMPLETE successful trajectories in the buffer.

THE ACTIONS ARE IN THE AGENT'S OWN ACTION SPACE. (steer, throttle), clipped to
the env's Box. Expert transitions are therefore ordinary off-policy rows: the
replay buffer and the HER relabeller cannot tell them apart, which is what
makes them usable.
"""
import os

import numpy as np

# Replan when the vehicle drifts this far from its planned route: the PID can
# be pushed off-course by traffic, and following a stale route afterwards
# produces exactly the wandering trajectories this is meant to avoid.
REPLAN_DISTANCE_M = 10.0
SAMPLING_RESOLUTION_M = 2.0

# PURE-PURSUIT AIM POINT, replacing a fixed lookahead of 3 nodes.
#
# THE BUG IT FIXES. `lookahead=3` at 2 m sampling aims 6 m ahead ALWAYS -- at
# any speed, through any curvature. A fixed aim point on a tight turn is the
# textbook way to cut a corner: the controller steers toward a point already
# past the apex, so the vehicle crosses the inside line. MEASURED with the
# fixed aim point: the oracle crossed a lane marking on 36% of episodes and
# scored 0.000 success.
#
# Standard pure pursuit makes the aim distance proportional to speed, so the
# controller looks further ahead when it is travelling faster and tightens up
# in slow corners. Clamped at both ends: below MIN the aim point is inside the
# vehicle and steering chatters, above MAX it is far enough to cut again.
#
# USES ONLY WHAT THE ORACLE ALREADY HAS -- its own speed (proprioception) and
# the route it planned. No new information source.
LOOKAHEAD_GAIN = float(os.environ.get("EXPERT_LOOKAHEAD_GAIN", 0.0))   # 0 = off
LOOKAHEAD_MIN_M = float(os.environ.get("EXPERT_LOOKAHEAD_MIN_M", 3.0))
LOOKAHEAD_MAX_M = float(os.environ.get("EXPERT_LOOKAHEAD_MAX_M", 12.0))

# SLOW DOWN FOR CURVATURE. 0 disables, leaving a constant target speed.
#
# The longitudinal PID holds one target regardless of the corner ahead, so the
# vehicle arrives at a turn carrying whatever the straight before it allowed.
# Scaling the target by the heading change over the next few route nodes is
# what a driver does and what BehaviorAgent does; the curvature comes from the
# planned route, which the oracle already holds.
CURVE_SLOWDOWN = float(os.environ.get("EXPERT_CURVE_SLOWDOWN", 0.0))
CURVE_LOOKAHEAD_NODES = int(os.environ.get("EXPERT_CURVE_NODES", 6))
CURVE_MIN_SPEED_FRAC = float(os.environ.get("EXPERT_CURVE_MIN_FRAC", 0.35))


class ExpertOracle:
    """Drives to the goal along the road network. One episode at a time."""

    last_plan_error = None

    def __init__(self, vehicle, world_map, action_space, lookahead=3,
                 target_speed_kmh=25.0, sampling_resolution=SAMPLING_RESOLUTION_M):
        import carla  # noqa: F401  (imported for type coherence at call sites)
        from src.agent_wrapper import VehiclePIDController
        from src.global_route_planner import GlobalRoutePlanner

        self._vehicle = vehicle
        self._map = world_map
        self._planner = GlobalRoutePlanner(world_map, sampling_resolution)
        dt = vehicle.get_world().get_settings().fixed_delta_seconds or 0.05
        self._pid = VehiclePIDController(vehicle=vehicle, dt=dt)

        self.action_space = action_space
        # A waypoint counts as PASSED only within half the route spacing.
        # MEASURED failure when this was REPLAN_DISTANCE_M (5 m) against 2 m
        # spacing: an 8.7 m route of 4 nodes had every node inside the
        # threshold at the start, so _advance discarded the whole route on its
        # first call and act() returned None 21 steps in, still 7.3 m short.
        # The pass threshold must be smaller than the node spacing or the route
        # deletes itself faster than the car drives it.
        self._pass_dist = max(0.5, float(sampling_resolution) * 0.5)
        self.lookahead = int(lookahead)
        self.sampling_resolution = float(sampling_resolution)
        self.target_speed = float(target_speed_kmh)
        self._route = []
        self._goal = None

    # -- planning ----------------------------------------------------------
    def reset(self, goal_location):
        """Plan a fresh route to `goal_location`. Returns route length in nodes."""
        self._goal = goal_location
        self._plan()
        return len(self._route)

    def _plan(self):
        origin = self._vehicle.get_transform().location
        self.last_plan_error = None
        try:
            self._route = self._planner.trace_route(origin, self._goal)
        except Exception as e:  # noqa: BLE001
            # A goal off the drivable graph has no route. Signalled by an empty
            # route rather than an exception so the caller can fall back to the
            # policy rather than aborting the run.
            #
            # The REASON is recorded. Swallowing it silently is what let a
            # completely broken planner (dijkstra_path called with a heuristic
            # kwarg, raising on every call) look like "the oracle just was not
            # picked" for 40 episodes.
            self.last_plan_error = f"{type(e).__name__}: {e}"
            self._route = []
        if not self._route and self.last_plan_error is None:
            d = origin.distance(self._goal) if self._goal is not None else -1.0
            self.last_plan_error = f"trace_route gave 0 waypoints (dist {d:.1f} m)"

    # -- control -----------------------------------------------------------
    def act(self):
        """One (steer, throttle) action, or None if no route is available."""
        if not self._route:
            return None

        tf = self._vehicle.get_transform()
        loc = tf.location
        self._advance(loc, tf.get_forward_vector())
        if not self._route:
            return None

        idx = self._aim_index()
        target_wp = self._route[idx][0]

        # Drifted off the plan (traffic, a failed turn) -- replan rather than
        # chase a waypoint that is no longer reachable from here.
        if target_wp.transform.location.distance(loc) > REPLAN_DISTANCE_M * 2:
            self._plan()
            if not self._route:
                return None
            target_wp = self._route[self._aim_index()][0]

        control = self._pid.run_step(self._speed_for_curve(), target_wp)
        return self._to_action(control)

    def _speed_mps(self):
        v = self._vehicle.get_velocity()
        return float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))

    def _aim_index(self):
        """Route index to steer at: pure pursuit when enabled, else fixed."""
        last = len(self._route) - 1
        if LOOKAHEAD_GAIN <= 0.0:
            return min(self.lookahead, last)
        d = LOOKAHEAD_GAIN * self._speed_mps()
        d = max(LOOKAHEAD_MIN_M, min(LOOKAHEAD_MAX_M, d))
        return min(max(1, int(round(d / max(self.sampling_resolution, 1e-6)))),
                   last)

    def _speed_for_curve(self):
        """Target speed, scaled down by the curvature of the route ahead.

        Heading change is summed over the next CURVE_LOOKAHEAD_NODES segments
        of the PLANNED route, so it sees the corner before the vehicle is in
        it. Floored at CURVE_MIN_SPEED_FRAC of the base target so the car never
        creeps to a stop on a long bend.
        """
        if CURVE_SLOWDOWN <= 0.0 or len(self._route) < 3:
            return self.target_speed
        pts = [w[0].transform.location
               for w in self._route[:CURVE_LOOKAHEAD_NODES + 1]]
        total = 0.0
        for i in range(len(pts) - 2):
            ax, ay = pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y
            bx, by = pts[i + 2].x - pts[i + 1].x, pts[i + 2].y - pts[i + 1].y
            na = float(np.hypot(ax, ay)); nb = float(np.hypot(bx, by))
            if na < 1e-6 or nb < 1e-6:
                continue
            cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
            total += float(np.arccos(cos))
        frac = 1.0 / (1.0 + CURVE_SLOWDOWN * total)
        return self.target_speed * max(CURVE_MIN_SPEED_FRAC, frac)

    def _advance(self, loc, fwd=None):
        """Drop route nodes the vehicle has genuinely passed.

        Two tests, and the second matters as much as the first: a node is
        passed if it is within half the spacing, OR if it lies BEHIND the
        vehicle. Distance alone cannot tell "just ahead" from "just behind",
        so on a tight turn the controller would keep steering back toward a
        waypoint it had already driven past.

        ALWAYS keeps at least one node: the final node is the goal, and the
        oracle must keep steering to it until the environment declares arrival
        rather than giving up a few metres short.
        """
        while len(self._route) > 1:
            wp = self._route[0][0].transform.location
            close = wp.distance(loc) < self._pass_dist
            behind = False
            if fwd is not None:
                behind = ((wp.x - loc.x) * fwd.x + (wp.y - loc.y) * fwd.y) < 0.0
            if close or behind:
                self._route.pop(0)
            else:
                break

    def _to_action(self, control):
        """PID output -> the agent's own action vector.

        VehiclePIDController.run_step returns a TUPLE (steering, acceleration),
        not a carla.VehicleControl -- its docstring claims it returns "distance
        in meters to the waypoint", which is wrong too. Assuming the object
        form cost a debugging cycle:
            AttributeError: 'tuple' object has no attribute 'brake'
        Both shapes are accepted so a future change to that controller cannot
        silently break the oracle.

        Brake, where it exists, folds into throttle=0 rather than becoming a
        third dimension: the policy's action space has none, and an expert
        acting outside that space would put rows in the buffer the actor could
        never reproduce.
        """
        if isinstance(control, (tuple, list, np.ndarray)):
            steer, throttle = float(control[0]), float(control[1])
        else:                                   # carla.VehicleControl
            steer = float(control.steer)
            # BRAKE IS SIGNED THROTTLE, NOT ZERO THROTTLE. The env's absolute
            # action space carries brake as a NEGATIVE a1 (compute_action:
            # a1 < 0 -> brake = -a1), so folding brake into throttle=0 threw
            # away a capability the env has. That mattered: at MAX_THROTTLE 0.4
            # this vehicle settles at 11.8 km/h, above BasicAgent's 9 km/h
            # target, so it brakes almost continuously -- and every brake became
            # a dead throttle. The old fold is kept behind EXPERT_FOLD_BRAKE=1
            # only so the two mappings can be A/B'd in bench_basic_agent.py.
            brake = float(getattr(control, "brake", 0.0) or 0.0)
            if os.environ.get("EXPERT_FOLD_BRAKE", "0") == "1":
                throttle = 0.0 if brake > 0.0 else float(control.throttle)
            else:
                throttle = float(control.throttle) - brake
        action = np.array([steer, throttle], np.float32)
        return np.clip(action, self.action_space.low,
                       self.action_space.high).astype(np.float32)

    @property
    def has_route(self):
        return bool(self._route)


def should_use_expert(rng, expert_prob):
    """Per-EPISODE coin flip. See module docstring on why not per step."""
    return bool(rng.random() < float(expert_prob))


def route_has_lane_change(route):
    """True when a [(waypoint, RoadOption)] route contains a lane change."""
    return any(str(opt).split(".")[-1] in ("CHANGELANELEFT", "CHANGELANERIGHT")
               for _, opt in route)


def lane_following_route(start_wp, goal_location, step_m=2.0, reach_m=3.0,
                         slack_m=40.0, stretch=3.0):
    """[(waypoint, RoadOption.LANEFOLLOW)] from start_wp to the goal using only
    waypoint.next() -- lane following, no lane changes -- or None.

    Depth-first over the branches next() offers at junctions, bounded by the
    straight-line distance times `stretch` plus `slack_m` of path (a route
    through a turn is much longer than the straight line), so it stays small:
    branches exist only at junctions and a 200 m goal crosses a handful."""
    import sys
    _p = "/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla"
    if _p not in sys.path:
        sys.path.append(_p)
    from agents.navigation.local_planner import RoadOption
    gx, gy = goal_location.x, goal_location.y

    def dist(wp):
        loc = wp.transform.location
        return float(np.hypot(loc.x - gx, loc.y - gy))

    max_steps = int((stretch * dist(start_wp) + slack_m) / step_m) + 1
    stack = [(start_wp, [start_wp])]
    while stack:
        wp, path = stack.pop()
        if dist(wp) <= reach_m:
            return [(w, RoadOption.LANEFOLLOW) for w in path]
        if len(path) >= max_steps:
            continue
        # nearest-first: push the branch closest to the goal LAST so it pops first
        nxt = sorted(wp.next(step_m), key=dist, reverse=True)
        for n in nxt:
            stack.append((n, path + [n]))
    return None


class BasicAgentExpert(ExpertOracle):
    """CARLA's own BasicAgent, wearing ExpertOracle's interface.

    WHY. Benchmarked on our exact task (Town02, goals 20-50 m, CLIPPED to our
    action space max_throttle 0.4 / max_steer 0.5), the reference agent reached
    the route end on 25 of 25 episodes with ZERO collisions, ZERO off-road and
    ZERO lane crossings, at ~10 km/h. Our hand-rolled wrapper over the same
    planner and PID was colliding on 36% and crossing a line on every turn. The
    difference is BasicAgent's LocalPlanner -- speed-aware waypoint purging
    rather than a fixed aim point -- and it is the reference implementation, so
    there is nothing of ours left to be wrong.

    THE ONE THING BasicAgent DOES NOT DO: reach the goal. It drives to the END
    OF ITS ROUTE, which the planner snaps to the nearest node -- up to one
    sampling resolution short of the requested point. Measured: 25/25 within
    5 m, only 2/25 within the env's 2 m arrival radius. So after done() this
    hands over to the inherited PID for a low-speed final approach to the EXACT
    goal location, which is what turns "reached the route end" into "arrived".

    Same contract as ExpertOracle: reset(goal) -> route length, act() -> action
    in the agent's own space or None, last_plan_error set on failure. Rows it
    produces are ordinary off-policy transitions; nothing here reaches the
    policy's observation.
    """

    FINAL_APPROACH_KMH = 6.0

    def __init__(self, vehicle, world_map, action_space, **kw):
        super().__init__(vehicle, world_map, action_space, **kw)
        import sys
        _p = "/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla"
        if _p not in sys.path:
            sys.path.append(_p)
        from agents.navigation.basic_agent import BasicAgent
        self._agent = BasicAgent(vehicle, target_speed=self.target_speed,
                                 map_inst=world_map)
        # n_vehicles is 0 and the env models neither lights nor signs, so
        # obeying them would only add stalls the policy never sees.
        self._agent.ignore_traffic_lights(True)
        self._agent.ignore_stop_signs(True)
        self._agent.ignore_vehicles(True)
        self._goal_loc = None
        self._planned = False

    def reset(self, goal_location):
        self.last_plan_error = None
        self._goal_loc = goal_location
        try:
            # PLAN FROM WHERE THE CAR IS NOW, NOT WHERE THE PLANNER LAST WAS.
            # BasicAgent.set_destination(goal) with no start_location traces
            # the route from the LocalPlanner's CURRENT target_waypoint -- the
            # waypoint the vehicle stood on when the agent was BUILT, or where
            # the previous episode ended. This expert is built once and the
            # env then TELEPORTS the hero every episode, so every route began
            # at a stale point tens of metres away: the first target sat
            # behind or beside the car (hard steer from standstill, crossing
            # within a metre) or far ahead on the old road (zero steer, slow
            # drift across the Broken line). That is the whole in-env
            # "weave": 9/9 oracle episodes lost, while bench_basic_agent.py --
            # which builds a fresh agent per episode -- drove 25/25 clean.
            # Passing start_location does not help either: that branch keeps
            # the old queue (clean_queue=False) and appends the new plan.
            lp = self._agent.get_local_planner()
            start_wp = self._map.get_waypoint(self._vehicle.get_location())
            end_wp = self._map.get_waypoint(goal_location)
            route = self._agent.trace_route(start_wp, end_wp)
            # NO LANE CHANGES (user 2026-09-28: "i want no lane change"). Every
            # goal lies on the start lane's own continuation (origin.next(d)),
            # yet CARLA's planner took a lane change on 6/600 Town05 routes
            # (scripts/goal_lane_audit.py) -- and the actor learns from what
            # the expert drives. Such a route is replaced by the lane-following
            # one, which always exists for these goals.
            if route_has_lane_change(route):
                lf = lane_following_route(start_wp, goal_location)
                if lf:
                    route = lf
                    print(f"[ExpertPlan] lane change in CARLA's route replaced "
                          f"by a lane-following one ({len(lf)} waypoints)",
                          flush=True)
            self._agent.set_global_plan(route, clean_queue=True)
            # PID memories are per-episode too: the derivative/integral
            # buffers otherwise carry the last episode's final error into the
            # first steps of this one.
            # (0.9.15 names them differently: longitudinal _error_buffer,
            # lateral _e_buffer.)
            ctrl = lp._vehicle_controller
            for c in (ctrl._lon_controller, ctrl._lat_controller):
                for attr in ("_error_buffer", "_e_buffer"):
                    buf = getattr(c, attr, None)
                    if buf is not None:
                        buf.clear()
            n = len(lp._waypoints_queue)
        except Exception as e:  # noqa: BLE001
            self.last_plan_error = f"{type(e).__name__}: {e}"
            n = 0
        # One line per plan: where the car is, where the route starts, how far
        # the two are apart. The stale-start bug above was only found by
        # printing exactly this. In its OWN try: a diagnostic that raises must
        # not turn a good plan into "empty route" (it did, in the unit test).
        try:
            tf = self._vehicle.get_transform()
            fw = tf.get_forward_vector()
            q = self._agent.get_local_planner()._waypoints_queue
            if n:
                r0 = q[0][0].transform.location
                dx, dy = r0.x - tf.location.x, r0.y - tf.location.y
                d0, ahead = float(np.hypot(dx, dy)), float(dx * fw.x + dy * fw.y)
            else:
                d0 = ahead = float("nan")
            print(f"[ExpertPlan] hero=({tf.location.x:.1f},{tf.location.y:.1f}) "
                  f"yaw={tf.rotation.yaw:.0f} route0_d={d0:.1f} ahead={ahead:+.1f} "
                  f"n={n} goal_d={tf.location.distance(goal_location):.1f}",
                  flush=True)
        except Exception:  # noqa: BLE001
            pass
        self._planned = n > 0
        if not self._planned and self.last_plan_error is None:
            self.last_plan_error = "BasicAgent produced an empty route"
        return n

    def act(self):
        if not self._planned:
            return None
        if not self._agent.done():
            return self._to_action(self._agent.run_step())

        # ROUTE END REACHED, GOAL NOT. The planner snaps the route's end to a
        # node up to one sampling step short of the goal, so crawl the last
        # metres ALONG THE LANE and stop once the goal is beside or behind
        # the car. It must never circle: the goal sits on the lane centre, so
        # a straight crawl passes within the env's 2 m arrival radius, and
        # steering at a point 2-3 m away at full lock does not -- measured in
        # explore mode, where nothing ends the episode at the goal, the old
        # PID approach hunted around the goal point and crossed a line on
        # 5/5 episodes at t~140-190, every one in this branch.
        tf = self._vehicle.get_transform()
        steer, go = self._final_approach(
            (tf.location.x, tf.location.y),
            (tf.get_forward_vector().x, tf.get_forward_vector().y),
            (self._goal_loc.x, self._goal_loc.y))
        if not go:
            return self._to_action((0.0, 0.0))
        import carla
        from types import SimpleNamespace
        target = SimpleNamespace(transform=carla.Transform(self._goal_loc))
        _, throttle = self._pid.run_step(self.FINAL_APPROACH_KMH, target)
        return self._to_action((steer, float(throttle)))

    # Steer limit while crawling: enough to track the lane centre, not enough
    # to turn around on it.
    FINAL_APPROACH_MAX_STEER = 0.15
    FINAL_APPROACH_STOP_M = 1.0

    @classmethod
    def _final_approach(cls, pos, forward, goal):
        """(steer, keep_going) for the crawl from the route's end to the goal.

        Pure so it can be tested: stop when the goal is within
        FINAL_APPROACH_STOP_M or no longer ahead of the car; otherwise steer
        gently toward it.
        """
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        fx, fy = forward
        ahead = dx * fx + dy * fy
        if dist < cls.FINAL_APPROACH_STOP_M or ahead <= 0.0:
            return 0.0, False
        # Signed lateral offset of the goal in the car's frame (left-handed
        # CARLA: positive cross = goal to the right = positive steer).
        cross = fx * dy - fy * dx
        steer = float(np.clip(np.arctan2(cross, ahead),
                              -cls.FINAL_APPROACH_MAX_STEER,
                              cls.FINAL_APPROACH_MAX_STEER))
        return steer, True
