"""Drive the REAL env stack with BasicAgentExpert and log every step.

bench_basic_agent.py showed 0/25 crossings in all four arms while the same
agent in-env crossed a line on 9/9 oracle episodes. This runs the exact
make_env + make_expert path the trainer uses, so whatever differs is here.
"""
import os
import sys

import numpy as np

sys.argv = [sys.argv[0]]
from train_carla_tqc import make_env, make_expert, latent_from_env  # noqa: E402

TOWN = os.environ.get("PROBE_TOWN", "Town01")
N_EP = int(os.environ.get("PROBE_EPISODES", 6))
PORT = int(os.environ.get("CARLA_PORT", 2000))

env = make_env(TOWN, 1234, latent_from_env(), port=PORT)
expert = make_expert(env, env.action_space)
# PROBE_MODE=0 forces explore mode, =1 goal mode, unset = env's own coin.
_pm = os.environ.get("PROBE_MODE")
if _pm is not None:
    w = env
    while w is not None and type(w).__name__ != "GatedReward":
        w = getattr(w, "env", None)
    assert w is not None, "no GatedReward in the stack"
    w.force_mode = int(_pm)
    print(f"[Probe] forcing mode={_pm}", flush=True)
WANT_MODE = int(_pm) if _pm is not None else 1
assert expert is not None, "no expert"
core = env.unwrapped.core
exp = env.unwrapped.experiment
print(f"[Probe] town={core.world.get_map().name} action_space={env.action_space} "
      f"delta={core.world.get_settings().fixed_delta_seconds}", flush=True)

last = {}
_orig = expert._to_action
def _rec(c):
    last["ctrl"] = c
    return _orig(c)
expert._to_action = _rec

def spd():
    v = core.hero.get_velocity()
    return 3.6 * float(np.hypot(v.x, v.y))

ep = 0
while ep < N_EP:
    obs, info = env.reset()
    if info.get("mode") != WANT_MODE:
        continue
    ep += 1
    goal = info["goal_location"]
    n = expert.reset(goal)
    hero0 = core.hero.get_transform().location
    _tf = core.hero.get_transform()
    _lp = expert._agent.get_local_planner()
    _q = list(_lp._waypoints_queue)
    _lwp = core.map.get_waypoint(_tf.location)
    print(f"  ids: core.hero={core.hero.id} agent.vehicle={expert._agent._vehicle.id} "
          f"expert.vehicle={expert._vehicle.id} | hero_yaw={_tf.rotation.yaw:.1f} "
          f"lane_wp_yaw={_lwp.transform.rotation.yaw:.1f} lane_id={_lwp.lane_id} road={_lwp.road_id}")
    _fw = _tf.get_forward_vector()
    for k, (w, opt) in enumerate(_q[:4]):
        dx, dy = w.transform.location.x - _tf.location.x, w.transform.location.y - _tf.location.y
        ahead = dx * _fw.x + dy * _fw.y
        print(f"  route[{k}] ({w.transform.location.x:.1f},{w.transform.location.y:.1f}) "
          f"d={np.hypot(dx,dy):.1f} ahead={ahead:+.1f} lane_id={w.lane_id} road={w.road_id} {opt}")
    _last = _q[-1][0].transform.location
    print(f"  route[-1] ({_last.x:.1f},{_last.y:.1f}) dist_to_goal={_last.distance(goal):.1f}")
    print(f"\n=== ep{ep} route_nodes={n} start=({hero0.x:.1f},{hero0.y:.1f}) "
          f"goal=({goal.x:.1f},{goal.y:.1f}) straight={hero0.distance(goal):.1f} m "
          f"mode={info.get('gated_mode', info.get('mode'))}", flush=True)
    for t in range(800):
        a = expert.act()
        if a is None:
            print(f"  t={t} expert returned None plan_error={expert.last_plan_error}"); break
        obs, r, term, trunc, info = env.step(a)
        c = last.get("ctrl")
        raw = (f"raw(st={c.steer:+.2f} th={c.throttle:.2f} br={c.brake:.2f})"
               if hasattr(c, "steer") else f"raw{tuple(round(float(x),2) for x in c)}")
        la = exp.last_action
        loc = core.hero.get_transform().location
        wp = core.map.get_waypoint(loc, project_to_road=False)
        lane_off = (None if wp is None else
                    float(np.hypot(loc.x - wp.transform.location.x,
                                   loc.y - wp.transform.location.y)))
        tw = expert._agent.get_local_planner().target_waypoint
        vt = expert._agent._vehicle.get_transform()
        twd = (None if tw is None else
               f"tgt=({tw.transform.location.x:.1f},{tw.transform.location.y:.1f}) "
               f"q={len(expert._agent.get_local_planner()._waypoints_queue)} "
               f"agent_sees=({vt.location.x:.1f},{vt.location.y:.1f},yaw{vt.rotation.yaw:.0f})")
        flag = f" {twd}"
        if info.get("lane_invasion_any"):
            flag = f"  <<< LANE {info.get('lane_marking_types')}"
        if info.get("off_road"):
            flag += "  <<< OFF_ROAD"
        if t % 10 == 0 or flag or term or trunc:
            print(f"  t={t:3d} a=({a[0]:+.2f},{a[1]:+.2f}) {raw} "
                  f"applied(st={la.steer:+.2f} th={la.throttle:.2f} br={la.brake:.2f}) "
                  f"v={spd():4.1f} pos=({loc.x:.1f},{loc.y:.1f}) lat_off={lane_off} "
                  f"junc={None if wp is None else wp.is_junction} "
                  f"rem={info.get('route_remaining_m')} r={r:+.2f}{flag}", flush=True)
        if term or trunc:
            why = {k: v for k, v in info.items()
                   if isinstance(v, (bool, int, float, str)) and k not in
                   ("hero_yaw",) and (v is True or k in ("is_success", "ttg_goals"))}
            print(f"  END t={t} term={term} trunc={trunc} {why}", flush=True)
            break
env.close()
