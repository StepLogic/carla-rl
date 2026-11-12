"""Expert recovery for c0's UNBROKEN evaluation-map drive.

User 2026-11-06: "i want one unbroke drive" -> "Expert recovers". The trace is a
single CARLA episode that the car is never moved out of. c0 drives (explore mode,
with the junction hops of torch_rl.junction_hops); at a fault -- a crossing,
leaving the road, a collision, or standing still -- CARLA's BasicAgent takes the
wheel WHERE THE CAR IS, drives it back into a lane, and hands back to c0 once the
car has been settled (in a lane, aligned with it, moving, fault-free) for a run
of steps. A car that cannot move at all under the expert ends the drive: that is
reported, never papered over with a teleport.

RecoveryGate is the pure state machine (tests/test_recovery.py); the caller
measures the lane offset, heading error and speed and owns both drivers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def wrap_deg(d: float) -> float:
    """Signed angle in [-180, 180)."""
    return (float(d) + 180.0) % 360.0 - 180.0


def best_lane(car_yaw: float, candidates):
    """Of (lane_yaw, lane) candidates, the lane the car points along -- after a
    crossing the nearest lane can be the OPPOSITE one, which the expert could
    only follow by turning round. None when there are none."""
    best, err = None, None
    for yaw, lane in candidates:
        e = abs(wrap_deg(car_yaw - yaw))
        if err is None or e < err:
            best, err = lane, e
    return best


@dataclass
class RecoveryGate:
    """Who drives: "c0", "expert", or "none" once the car is wedged."""

    min_steps: int = 20          # the expert drives at least this long (2 s)
    settle_steps: int = 10       # ... and hands back after this many settled steps
    max_lat_m: float = 0.6       # settled: within this of the lane centre
    max_yaw_deg: float = 15.0    # ... aligned with the lane within this
    min_kmh: float = 5.0         # ... and moving at least this fast
    stall_steps: int = 50        # c0 standing still this long (5 s) is a fault
    blocked_steps: int = 15      # the expert not moving for 1.5 s: it is blocked
    backup_steps: int = 20       # ... so the car REVERSES for 2 s, then re-plans
    max_backups: int = 5         # ... at most this many times per takeover
    wedged_steps: int = 300      # the expert unable to move for 30 s ends the drive

    driver: str = "c0"
    events: List[dict] = field(default_factory=list)
    _since: int = 0
    _settled: int = 0
    _still: int = 0
    _backing: int = 0
    _backups: int = 0

    def update(self, t: int, fault: Optional[str], lat_m: float, yaw_err_deg: float,
               speed_kmh: float) -> Optional[str]:
        """One step's measurements -> "takeover", "backup", "replan", "handback",
        "wedged" or None. "backup": start reversing (the env has no reverse in its
        action space, so the caller drives it, see `backing`); "replan": the
        reverse is over, plan the expert's lane again from where the car is."""
        if self.driver == "none":
            return None
        moving = float(speed_kmh) >= 0.5
        self._still = 0 if moving else self._still + 1
        if self.driver == "c0":
            if fault is None and self._still >= self.stall_steps:
                fault = "stalled"
            if fault is None:
                return None
            self.driver, self._since, self._settled, self._still = "expert", 0, 0, 0
            self._backing, self._backups = 0, 0
            self.events.append({"fault": fault, "t": int(t), "steps": 0, "end": None,
                                "backups": 0})
            return "takeover"
        # the expert has the wheel
        self._since += 1
        ev = self.events[-1]
        ev["steps"] = self._since
        if self._backing:
            # reversing out: neither settling nor stillness counts meanwhile
            self._backing -= 1
            self._still, self._settled = 0, 0
            return "replan" if self._backing == 0 else None
        if self._still >= self.blocked_steps and self._backups < self.max_backups:
            self._backups += 1
            ev["backups"] = self._backups
            self._backing, self._still, self._settled = self.backup_steps, 0, 0
            return "backup"
        if self._still >= self.wedged_steps:
            self.driver, ev["end"] = "none", "wedged"
            return "wedged"
        settled = (fault is None and abs(float(lat_m)) <= self.max_lat_m
                   and abs(wrap_deg(yaw_err_deg)) <= self.max_yaw_deg
                   and float(speed_kmh) >= self.min_kmh)
        self._settled = self._settled + 1 if settled else 0
        if self._since >= self.min_steps and self._settled >= self.settle_steps:
            self.driver, ev["end"] = "c0", "handback"
            self._still = 0
            return "handback"
        return None

    @property
    def backing(self) -> bool:
        """True while the car is reversing out of a block."""
        return self._backing > 0

    def summary(self) -> Dict[str, object]:
        by: Dict[str, int] = {}
        for ev in self.events:
            by[ev["fault"]] = by.get(ev["fault"], 0) + 1
        return {"takeovers": len(self.events), "by_fault": by,
                "expert_steps": int(sum(ev["steps"] for ev in self.events)),
                "backups": int(sum(ev.get("backups", 0) for ev in self.events)),
                "wedged": any(ev["end"] == "wedged" for ev in self.events)}
