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
import numpy as np

# Replan when the vehicle drifts this far from its planned route: the PID can
# be pushed off-course by traffic, and following a stale route afterwards
# produces exactly the wandering trajectories this is meant to avoid.
REPLAN_DISTANCE_M = 10.0
SAMPLING_RESOLUTION_M = 2.0


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

        idx = min(self.lookahead, len(self._route) - 1)
        target_wp = self._route[idx][0]

        # Drifted off the plan (traffic, a failed turn) -- replan rather than
        # chase a waypoint that is no longer reachable from here.
        if target_wp.transform.location.distance(loc) > REPLAN_DISTANCE_M * 2:
            self._plan()
            if not self._route:
                return None
            target_wp = self._route[min(self.lookahead, len(self._route) - 1)][0]

        control = self._pid.run_step(self.target_speed, target_wp)
        return self._to_action(control)

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
            throttle = (0.0 if getattr(control, "brake", 0.0) > 0.0
                        else float(control.throttle))
        action = np.array([steer, throttle], np.float32)
        return np.clip(action, self.action_space.low,
                       self.action_space.high).astype(np.float32)

    @property
    def has_route(self):
        return bool(self._route)


def should_use_expert(rng, expert_prob):
    """Per-EPISODE coin flip. See module docstring on why not per step."""
    return bool(rng.random() < float(expert_prob))
