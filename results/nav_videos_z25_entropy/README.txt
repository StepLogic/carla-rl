Image-goal navigation with an entropy-based topological map in CARLA (held-out Town02)

Policy: imitation (behaviour cloning) policy, trained only in Town01/04/05/10HD.
Map: entropy map built from one expert drive of Town02; key nodes where the policy's
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

Files:
  entropy_nav_Town02_seed41012_reached.mp4   143 m route, reached
  entropy_nav_Town02_seed41026_reached.mp4   141 m route, reached
  entropy_nav_Town02_seed41115_reached.mp4   136 m route, reached
  entropy_nav_Town02_seed41008_reached.mp4    95 m route, reached
  entropy_nav_Town02_seed41044_reached.mp4    91 m route, reached
  entropy_nav_Town02_seed41001_off_road.mp4  106 m route, failed (left the road)

These six were chosen to illustrate successes and a failure; they are not a random
sample. The method's success rate on the full hard set is 0.13 (Town02 0.18).
