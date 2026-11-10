Image-goal navigation with an entropy-based topological map in CARLA (held-out Town03)

Policy: imitation (behaviour cloning) policy, trained only in Town01/04/05/10HD.
Map: entropy map built from one expert drive of Town03; key nodes where the policy's
action entropy is a robust outlier (online MAD, z = 2.5), plus image-coverage gap nodes.
Localisation and subgoal arrival use camera images only (no pose).
Episodes: hard set, 50-150 m routes with at least two junction turns.
Success: reaching within 1 m of the goal.

Each video (2x real time):
  left top     car camera
  left bottom  image the policy is steering toward (current waypoint, then the goal)
  right        overhead view: shortest road route (blue), map waypoints on the plan
               (orange), current waypoint (magenta ring), car track (red), start (green),
               goal (yellow star)

ALL SIX EPISODES FAIL in these recordings. Four of them (41028, 41123, 41176, 41189)
reached the goal in the original evaluation; none reproduced on this rerun. Town03 is
where every method struggles: the entropy map's hard-set success there is 0.07
(7 of 100 episodes), and at that level individual successes do not reproduce reliably,
because CARLA runs are not exactly repeatable and a lane crossing on Town03's narrow
lanes ends an episode early.

Files (route length, how it ended, closest approach to the goal):
  entropy_nav_Town03_seed41189_clock.mp4      85 m  ran out of time   3.6 m  <- the instructive one
  entropy_nav_Town03_seed41176_crossing.mp4   90 m  crossed a line   52.7 m
  entropy_nav_Town03_seed41010_crossing.mp4  138 m  crossed a line   51.2 m
  entropy_nav_Town03_seed41028_crossing.mp4   81 m  crossed a line   47.5 m
  entropy_nav_Town03_seed41123_off_road.mp4  144 m  left the road    65.0 m
  entropy_nav_Town03_seed41016_crossing.mp4  146 m  crossed a line   86.0 m

Seed 41189: the plan is correct and the car follows all 7 waypoints round the
roundabout, but it passes the goal (closest 3.6 m, at 17 km/h, step 213) two steps
BEFORE the last waypoint's image arrival fires and the goal image is handed over
(step 215). It then drives away from the goal until the time limit. This is a
navigation-timing failure, not a planning one.

The other five never come within 47 m of the goal: the policy crosses a lane line or
leaves the road early, which is the driving skill failing, not the map.
