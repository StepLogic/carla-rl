"""The 16-slot visitation ring from the Habitat peer's gated explorer (v1).

Ported to match `explore_mode_wrapper.py` in robot-learning-tools, checkpoint
`robot_policy_pointgoal_gated_s42_snap/checkpoint_600000`. Their spec, verbatim:

    bearings b_k = -pi + k*2pi/16, k = 0..15, full circle in the ROBOT frame.
    For each bearing, sample ranges r = 0.5, 1.0, ..., 3.0 m along that bearing
    from the current pose, and output the FRACTION of those 6 sample points
    whose cell is in the visited set. Values in [0, 1], binary membership,
    NOT counts. Visited set resets at every episode start.

WHAT THE RING IS FOR. In their v1 the depth ring does obstacle avoidance and
this ring does the exploring -- it is the only thing telling the policy which
directions it has already been. We drop their depth ring on their advice (the
frozen VAE latent over a semantic camera already carries obstacle information),
so this ring is the whole of the explore-specific observation.

THEIR 180-DEGREE FRAME BUG DOES NOT PORT. v1 was trained with these bearings
rotated 180 deg relative to its depth ring, and they kept it because it still
worked (VISIT_RING_LEGACY=1 reproduces it). That offset was a MISALIGNMENT
BETWEEN TWO RINGS. We have only one ring, so there is nothing to be misaligned
with and no bug to reproduce -- an arbitrary 180 deg rotation of a full circle
of bearings is just a relabelling of slots. Reproducing it here would be cargo
cult, so we do not.

CARLA IS LEFT-HANDED, AND THIS IS WHERE THE REAL BUG LIVES.

    forward = ( cos yaw,  sin yaw)
    right   = (-sin yaw,  cos yaw)

A textbook right-handed rotation gives a MIRRORED bearing here. That exact
error survived in this repo's goal block for an entire training run: a goal 20 m
to the right reported left = +20.0 when the truth is -20.0, so the policy was
told to steer away from the goal at every turn. It looked like "drives but never
arrives" rather than anything obviously broken. The ring below rotates bearings
into the world with the same left-handed convention, and the tests pin it.
"""
import numpy as np

N_SLOTS = 16
# 0.5, 1.0, ... 3.0 m -- their six sample ranges, unchanged.
RING_RANGES_M = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
CELL_SIZE_M = 0.5

# b_k = -pi + k * 2pi/16, exactly as specified.
BEARINGS = np.array([-np.pi + k * 2.0 * np.pi / N_SLOTS for k in range(N_SLOTS)])


def cell_of(pos, cell_size=CELL_SIZE_M):
    """Ground-plane cell index. Their `cell_of(pos) = (floor(x/0.5), floor(z/0.5))`.

    Habitat's ground plane is (x, z); CARLA's is (x, y). Same operation on
    whichever pair the simulator calls horizontal.
    """
    p = np.asarray(pos, np.float64)
    return (int(np.floor(p[0] / cell_size)), int(np.floor(p[1] / cell_size)))


def visit_ring(hero_pos, hero_yaw, visited, cell_size=CELL_SIZE_M,
               ranges=RING_RANGES_M):
    """-> (16,) float32 in [0, 1]: fraction of sampled points already visited.

    `visited` is a set of cell tuples as produced by `cell_of`.
    """
    hero = np.asarray(hero_pos, np.float64)
    cy, sy = np.cos(float(hero_yaw)), np.sin(float(hero_yaw))
    out = np.zeros(N_SLOTS, np.float32)
    for k, b in enumerate(BEARINGS):
        # Bearing b in the ROBOT frame -> world, left-handed (see module docstring).
        # A point at bearing b, range r is r*(cos b) along forward plus
        # r*(sin b) along LEFT; left = -right = (sin yaw, -cos yaw).
        fx, fy = cy, sy
        lx, ly = sy, -cy
        cb, sb = np.cos(b), np.sin(b)
        hits = 0
        for r in ranges:
            px = hero[0] + r * (cb * fx + sb * lx)
            py = hero[1] + r * (cb * fy + sb * ly)
            if cell_of((px, py), cell_size) in visited:
                hits += 1
        out[k] = hits / float(len(ranges))
    return out
