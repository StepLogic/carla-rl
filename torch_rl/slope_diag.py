"""Episode transforms for the offline critic-slope diagnosis (user 2026-11-07,
"Diagnose the slope first"; scripts/offline_critic_slope.py).

Each variant removes ONE candidate cause of the warm-up critic's
"steer left + less throttle beats the expert" slope from the recorded episodes
before they fill the replay (expert_data.fill_buffer's `transform`): a function
episode -> episode, or None to leave the episode out. The loaded episode is never
modified in place.
"""
from functools import partial

# Steps cut from the end of an episode that reached its goal: the braking
# approach (arrival needs < 5 km/h inside the radius, from ~15 km/h) and the
# +100 arrival bonus on the last step. ~3 s at the env's control rate.
ARRIVAL_CUT = 30

_PER_STEP = ("act_exec", "act_label", "reward", "speed", "pos")   # T rows
_PER_FRAME = ("frames", "binoc")                                     # T + 1 rows


def cut_arrival(ep, n=ARRIVAL_CUT):
    """An episode that ended 'reached' without its last `n` transitions, ending
    TRUNCATED (its new last state is not terminal); None if it is not longer
    than `n`. Any other ending is returned unchanged (the same object)."""
    if ep["end"] != "reached":
        return ep
    T = len(ep["act_exec"])
    if T <= n:
        return None
    out = dict(ep)
    for k in _PER_STEP:
        out[k] = ep[k][:T - n]
    for k in _PER_FRAME:
        out[k] = ep[k][:T - n + 1]
    out["terminal"], out["truncated"] = False, True
    return out


def expert_driven(ep):
    """The episode if the expert drove it, else None (DAgger rounds record the
    clone driving with the expert's labels: driver 'policy')."""
    return ep if ep["driver"] == "expert" else None


VARIANTS = {
    "baseline": None,
    "no_arrival": partial(cut_arrival, n=ARRIVAL_CUT),
    "expert_only": expert_driven,
}
