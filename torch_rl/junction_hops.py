"""Junction hops for c0's evaluation-map trace.

User 2026-11-06: "Can we explicitly random actions at junctions" -> "Goal hop per
junction". c0 drives the trace in EXPLORE mode; nearing a junction it is handed a
goal image 10-30 m past a RANDOMLY chosen exit
(vae_nav.goal_chain.goal_past_next_junction) and drives there in GOAL mode, then
explores again. Node placement still uses the explore-mode entropy, so the hop is
only there to push the drive down exits the explore policy would never take.

This module holds the PURE part of that schedule -- when a hop starts, when the
goal view counts as rendered, how a hop ends, which exit it asked for and which it
drove -- so it is testable without CARLA (tests/test_junction_hops.py). The caller
owns the camera, the policy and the recording; it only has to:

    if sched.want_hop(ahead_m, in_junction):
        sched.begin(Hop(next_id, goal_xy=..., goal_yaw=..., start_t=t,
                        start_xy=car_xy, start_yaw=car_yaw))
    if sched.mode == "pending":
        sched.goal_view_ready(goal_camera_pos)      # None until the camera moved
    elif sched.mode == "goal":
        sched.step(t, car_xy, car_yaw)              # returns the end reason or None
    if sched.take_reset():
        hidden = None                               # soft reset: car stays, GRU fresh

Every mode switch (explore -> goal and goal -> explore) is a soft reset, as the
goal chain's switches are: take_reset() is True exactly once per switch.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# A turn of at least this many degrees is a turn, not a straight-through exit.
TURN_DEG = 30.0
# How close the rendered goal camera must be to the asked-for pose to count.
GOAL_VIEW_M = 2.0


def _wrap(deg: float) -> float:
    """Signed yaw difference in (-180, 180]."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def exit_class(start_yaw: float, goal_yaw: float) -> str:
    """Which exit a yaw change takes: "left", "right" or "straight".

    CARLA's yaw is left-handed seen from above (z into the ground), so a POSITIVE
    yaw change is a RIGHT turn.
    """
    d = _wrap(goal_yaw - start_yaw)
    if abs(d) < TURN_DEG:
        return "straight"
    return "right" if d > 0.0 else "left"


@dataclass
class Hop:
    """One junction hop: the goal asked for, and how it turned out."""

    id: int
    goal_xy: Tuple[float, float]
    goal_yaw: float
    start_t: int
    start_xy: Tuple[float, float]
    start_yaw: float
    planned_exit: str = ""
    driven_exit: Optional[str] = None
    end: Optional[str] = None
    end_t: Optional[int] = None
    goal_t: Optional[int] = None          # step the goal mode actually started
    closest_m: Optional[float] = None     # closest approach to the goal

    def __post_init__(self) -> None:
        if not self.planned_exit:
            self.planned_exit = exit_class(self.start_yaw, self.goal_yaw)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


@dataclass
class HopSchedule:
    """The explore -> goal -> explore schedule of one trace segment's hops."""

    trigger_m: float = 20.0          # a junction this close ahead starts a hop
    arrival_m: float = 2.0           # within this of the goal: reached
    miss_m: float = 10.0             # this far back past the closest approach: missed
    max_hop_steps: int = 600         # a hop may not outlast this many env steps
    max_pending_steps: int = 20      # steps to wait for the goal camera
    goal_view_m: float = GOAL_VIEW_M

    mode: str = "explore"            # "explore" | "pending" | "goal"
    hop: Optional[Hop] = None
    done: List[Hop] = field(default_factory=list)
    _pending_steps: int = 0
    _reset: bool = False

    # ------------------------------------------------------------------ queries
    @property
    def mode_bit(self) -> int:
        """The observation's mode bit: 1 only while actually driving a goal."""
        return 1 if self.mode == "goal" else 0

    def want_hop(self, ahead_m: Optional[float], in_junction: bool) -> bool:
        """Start a hop only while exploring, with a junction close AHEAD of us."""
        if self.mode != "explore" or in_junction or ahead_m is None:
            return False
        return float(ahead_m) <= self.trigger_m

    def take_reset(self) -> bool:
        """True exactly once after each mode switch (the GRU's soft reset)."""
        was, self._reset = self._reset, False
        return was

    # ----------------------------------------------------------------- the hop
    def begin(self, hop: Hop) -> None:
        """Ask for a hop: the goal is chosen, the goal camera has not rendered yet."""
        self.hop = hop
        self.mode = "pending"
        self._pending_steps = 0

    def goal_view_ready(self, goal_pos: Optional[Tuple[float, float]]) -> bool:
        """Has the goal camera reached the pose we asked for?

        Until it has, the hop waits; after max_pending_steps it is abandoned and the
        drive goes back to exploring, so a goal the camera cannot reach never stalls
        the trace.
        """
        if self.mode != "pending" or self.hop is None:
            return False
        if goal_pos is not None and _dist(goal_pos, self.hop.goal_xy) <= self.goal_view_m:
            self.mode = "goal"
            self._reset = True                       # fresh GRU for the goal mode
            self.hop.closest_m = None
            return True
        self._pending_steps += 1
        if self._pending_steps >= self.max_pending_steps:
            self._close("no_goal_view", t=self.hop.start_t + self._pending_steps)
        return False

    def step(self, t: int, pos: Tuple[float, float], yaw: float) -> Optional[str]:
        """One goal-mode step. Returns the end reason, or None while the hop runs."""
        if self.mode != "goal" or self.hop is None:
            return None
        d = _dist(pos, self.hop.goal_xy)
        best = self.hop.closest_m
        if d <= self.arrival_m:
            return self._close("reached", t=t, yaw=yaw)
        if best is not None and d >= best + self.miss_m:
            return self._close("missed", t=t, yaw=yaw)
        if t - self.hop.start_t >= self.max_hop_steps:
            return self._close("timeout", t=t, yaw=yaw)
        self.hop.closest_m = d if best is None else min(best, d)
        return None

    def segment_end(self, t: int, reason: str) -> Optional[str]:
        """The segment ended under the hop (a fault or the cap): close it with that."""
        if self.mode == "explore" or self.hop is None:
            return None
        return self._close(reason, t=t)

    def _close(self, end: str, t: int, yaw: Optional[float] = None) -> str:
        hop = self.hop
        hop.end = end
        hop.end_t = t
        if yaw is not None:
            hop.driven_exit = exit_class(hop.start_yaw, yaw)
        self.done.append(hop)
        self.hop = None
        self.mode = "explore"
        self._pending_steps = 0
        self._reset = True                            # fresh GRU for exploring again
        return end

    # ---------------------------------------------------------------- reporting
    def summary(self) -> Dict[str, object]:
        """Counts for the trace's index.json: how many hops, how they ended, which
        exits they asked for, and which exits were actually reached."""
        ends: Dict[str, int] = {}
        planned: Dict[str, int] = {}
        reached: Dict[str, int] = {}
        for hop in self.done:
            planned[hop.planned_exit] = planned.get(hop.planned_exit, 0) + 1
            if hop.end is not None:
                ends[hop.end] = ends.get(hop.end, 0) + 1
            if hop.end == "reached":
                reached[hop.planned_exit] = reached.get(hop.planned_exit, 0) + 1
        return {
            "attempted": len(self.done),
            "ends": ends,
            "planned_exits": planned,
            "reached_by_exit": reached,
        }
