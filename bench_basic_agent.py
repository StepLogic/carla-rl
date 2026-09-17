"""Can CARLA's OWN reference agent drive our task? The solvability floor.

WHY THIS EXISTS. Our ExpertOracle is a hand-rolled wrapper over CARLA's
GlobalRoutePlanner and VehiclePIDController -- it does NOT use BasicAgent or
LocalPlanner. When it fails we cannot tell whether the task is hard or our
wrapper is bad. BasicAgent is the reference implementation shipped with the
simulator; if IT cannot drive the task, the task is the problem.

TWO ARMS, AND THE PAIR IS THE POINT:

    stock       BasicAgent's own limits (max_throttle 0.75, max_steer 0.8).
                Answers: is this task doable AT ALL?
    clipped     our action space (MAX_THROTTLE, MAX_STEER), so the control is
                squeezed through exactly the box our policy acts in.
                Answers: is it doable UNDER OUR CONSTRAINTS?

A gap between the two is a cost our action space imposes, not a driving
failure -- and that distinction has been confounded all evening.

MEASURED THE SAME WAY THE ENV MEASURES. Real lane-invasion and collision
sensors, and the same off-road probe (get_waypoint(project_to_road=False) is
None), so the numbers are comparable to [Why] rather than merely similar.
Marking types are recorded too, since only NONE and Broken have ever fired here
and the distinction decides whether a crossing is "illegal" at all.

Goals are sampled as reset_hero does: walk forward along the lane by a distance
drawn from the band, and (as GOAL_NOT_IN_JUNCTION does) reject a goal landing
inside a junction.
"""
import argparse
import math
import os
import random
import sys

sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
import carla                                          # noqa: E402
from agents.navigation.basic_agent import BasicAgent   # noqa: E402


def speed_kmh(v):
    vel = v.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--goal-min", type=float, default=20.0)
    ap.add_argument("--goal-max", type=float, default=50.0)
    ap.add_argument("--max-steps", type=int, default=750)
    ap.add_argument("--target-speed", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1234)
    # 0 = BasicAgent's own limits. Otherwise squeeze its control through ours.
    ap.add_argument("--max-throttle", type=float, default=0.0)
    ap.add_argument("--max-steer", type=float, default=0.0)
    # DISCRIMINATING ARMS. In-env the same agent weaves on straight road
    # (Broken@road 0.50, NONE@road 0.47) while this bench showed 0/25 crossings.
    # Two things the bench did not replicate, each switchable:
    #   --env-action-path  route the control through the env's mapping:
    #                      steer clipped, throttle clipped to [MIN_THROTTLE,
    #                      max], brake carried as NEGATIVE throttle
    #                      (--fold-brake instead reproduces the OLD _to_action,
    #                      which turned every brake into throttle=0)
    #   --junction-origin  place the origin one goal-distance BEFORE a junction
    #                      so the route turns through it, as reset_hero does at
    #                      JUNCTION_GOAL_FRAC=0.7
    ap.add_argument("--env-action-path", action="store_true")
    ap.add_argument("--fold-brake", action="store_true")
    ap.add_argument("--min-throttle", type=float, default=0.0)
    ap.add_argument("--junction-origin", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)

    client = carla.Client("localhost", args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.town)
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.1
    world.apply_settings(s)
    amap = world.get_map()
    bp = world.get_blueprint_library()
    veh_bp = bp.find("vehicle.micro.microlino")
    # Non-junction waypoints, exactly the pool reset_hero draws its origin from.
    waypoints = [w for w in amap.generate_waypoints(2.0) if not w.is_junction]
    junctions = [w for w in amap.generate_waypoints(2.0) if w.is_junction]

    lane_bp = bp.find("sensor.other.lane_invasion")
    coll_bp = bp.find("sensor.other.collision")

    clipped = args.max_throttle > 0.0
    if args.env_action_path and not clipped:
        args.max_throttle, args.max_steer = 0.4, 0.5
    print(f"[Bench] BasicAgent on {args.town}, {args.episodes} episodes, "
          f"goals {args.goal_min:g}-{args.goal_max:g} m, target {args.target_speed:g} km/h, "
          f"{'CLIPPED to our action space' if clipped else 'STOCK limits'}", flush=True)

    stats = {"arrived": 0, "lane": 0, "coll": 0, "offroad": 0, "timeout": 0,
             "within2": 0}
    marks, speeds, steps_used = {}, [], []

    for ep in range(args.episodes):
        origin = None
        if args.junction_origin and junctions:
            jwp = random.choice(junctions)
            back = random.uniform(args.goal_min, args.goal_max)
            for d in (back, back * 0.75, back * 0.5, 10.0, 5.0):
                prev = jwp.previous(float(d))
                if prev:
                    origin = random.choice(prev)
                    break
        if origin is None:
            origin = random.choice(waypoints)
        goal_wp = None
        for _ in range(40):
            d = random.uniform(args.goal_min, args.goal_max)
            cand = origin.next(d)
            if cand and not cand[-1].is_junction:
                goal_wp = cand[-1]
                break
        if goal_wp is None:
            continue

        tf = origin.transform
        # +1.0, MATCHING reset_hero. At +0.5 the microlino spawns intersecting
        # the road surface: physics resolves the overlap by shoving the car,
        # which showed up as 40% collisions, 60% off-road and a 0.1 km/h median
        # speed -- a harness artefact that would have been read as BasicAgent
        # failing the task.
        tf.location.z += 1.0
        vehicle = world.try_spawn_actor(veh_bp, tf)
        if vehicle is None:
            continue
        # SETTLE BEFORE ATTACHING THE SENSORS. The vehicle is spawned 1 m above
        # the road (as reset_hero does) and DROPS onto it. That landing fires the
        # collision sensor, and briefly leaves the car off any lane polygon, so
        # sensors attached earlier record a collision and an off-road on EVERY
        # episode -- these flags are sticky, so the whole episode is then marked.
        # Measured with sensors attached first: collision 0.84, off_road 0.80.
        # The env avoids this because it recomputes `collision` per step from a
        # buffer cleared each tick, rather than latching it.
        for _ in range(10):
            world.tick()

        ev = {"lane": False, "coll": False, "marks": set()}
        lane_s = world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle)
        coll_s = world.spawn_actor(coll_bp, carla.Transform(), attach_to=vehicle)

        def on_lane(e, ev=ev):
            ev["lane"] = True
            for m in e.crossed_lane_markings:
                ev["marks"].add(str(m.type).split(".")[-1])

        lane_s.listen(on_lane)
        coll_s.listen(lambda e, ev=ev: ev.update(coll=True))

        agent = BasicAgent(vehicle, target_speed=args.target_speed, map_inst=amap)
        # No traffic is spawned (n_vehicles: 0), so these only remove stalls at
        # lights the env does not model either.
        agent.ignore_traffic_lights(True)
        agent.ignore_stop_signs(True)
        agent.ignore_vehicles(True)      # n_vehicles: 0, so nothing to yield to
        agent.set_destination(goal_wp.transform.location)

        outcome, offroad = "timeout", False
        for step in range(args.max_steps):
            world.tick()
            ctrl = agent.run_step()
            if args.env_action_path:
                # Exactly what _to_action + compute_action (absolute) do.
                st = max(-args.max_steer, min(args.max_steer, float(ctrl.steer)))
                if args.fold_brake:
                    a1 = 0.0 if ctrl.brake > 0.0 else float(ctrl.throttle)
                else:
                    a1 = float(ctrl.throttle) - float(ctrl.brake)
                if a1 >= 0.0:
                    ctrl = carla.VehicleControl(
                        steer=st, throttle=max(args.min_throttle,
                                               min(a1, args.max_throttle)),
                        brake=0.0)
                else:
                    ctrl = carla.VehicleControl(steer=st, throttle=0.0,
                                                brake=min(-a1, 1.0))
            elif clipped:
                ctrl.throttle = min(ctrl.throttle, args.max_throttle)
                ctrl.steer = max(-args.max_steer, min(args.max_steer, ctrl.steer))
            vehicle.apply_control(ctrl)
            speeds.append(speed_kmh(vehicle))

            loc = vehicle.get_transform().location
            if amap.get_waypoint(loc, project_to_road=False) is None:
                offroad = True
            if loc.distance(goal_wp.transform.location) < 2.0:
                outcome = "arrived"
                break
            if agent.done():
                # The planner snaps the route's end to the nearest node, up to
                # one sampling resolution short of the requested point, so a
                # driver that finishes its route stops ~2-5 m from the goal.
                # Report BOTH radii: the env's 2 m arrival test, and a 5 m
                # "reached the route's end" test that does not penalise the
                # snap. The gap between them is the planner's, not the driver's.
                outcome = "arrived" if loc.distance(
                    goal_wp.transform.location) < 5.0 else "route_end"
                break

        if os.environ.get("BENCH_VERBOSE"):
            _d = loc.distance(goal_wp.transform.location)
            print(f"  ep{ep:02d} {outcome:9s} steps={step+1:4d} "
                  f"dist_to_goal={_d:5.1f} speed={speed_kmh(vehicle):5.1f} "
                  f"lane={ev['lane']} coll={ev['coll']} offroad={offroad} "
                  f"route_left={len(agent.get_local_planner()._waypoints_queue)}",
                  flush=True)
        stats[{"arrived": "arrived"}.get(outcome, "timeout")] += 1
        if loc.distance(goal_wp.transform.location) < 2.0:
            stats["within2"] += 1
        if ev["lane"]:
            stats["lane"] += 1
        if ev["coll"]:
            stats["coll"] += 1
        if offroad:
            stats["offroad"] += 1
        for m in ev["marks"]:
            marks[m] = marks.get(m, 0) + 1
        steps_used.append(step + 1)

        lane_s.destroy(); coll_s.destroy(); vehicle.destroy()

    n = max(1, sum(1 for _ in steps_used))
    print(f"\n[Bench] n={n}")
    print(f"  arrived <5m  {stats['arrived'] / n:.3f}   (route end reached)")
    print(f"  arrived <2m  {stats['within2'] / n:.3f}   (the env's arrival radius)")
    print(f"  lane_any     {stats['lane'] / n:.3f}")
    print(f"  collision    {stats['coll'] / n:.3f}")
    print(f"  off_road     {stats['offroad'] / n:.3f}")
    print(f"  mean steps   {sum(steps_used) / n:.0f} of {args.max_steps}")
    if speeds:
        speeds.sort()
        print(f"  speed median {speeds[len(speeds) // 2]:.1f} km/h  "
              f"max {speeds[-1]:.1f}")
    print(f"  markings     {marks}")


if __name__ == "__main__":
    main()
