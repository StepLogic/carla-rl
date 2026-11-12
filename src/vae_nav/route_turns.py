"""Junction-turn directions on a planner route (junction bands, 2026-11-12).

The probe (train_carla_sac._route_turns) counts a junction TURN as a run of RoadOption
LEFT/RIGHT on the 2 m GlobalRoutePlanner route. turn_runs applies the same rule and also
records each run's direction ("L" or "R", from the run's first option), so offline
reconstructions can be checked against the probe's stored count. A run that holds both
LEFT and RIGHT counts once, as the probe counts it, and is flagged in mixed_runs.
"""

TURNS = ("LEFT", "RIGHT")


def _name(opt):
    return getattr(opt, "name", opt)


def turn_runs(options):
    """options: RoadOption values (or their names) along a route, in order.
    -> {"seq": e.g. "LR", "turns": len(seq), "mixed_runs": runs holding both L and R}"""
    seq, mixed = [], 0
    run_first, run_mixed = None, False
    for opt in options:
        n = _name(opt)
        if n in TURNS:
            if run_first is None:                   # a new run: one turn
                run_first, run_mixed = n, False
                seq.append(n[0])
            elif n != run_first and not run_mixed:
                run_mixed = True
                mixed += 1
        else:
            run_first = None
    return {"seq": "".join(seq), "turns": len(seq), "mixed_runs": mixed}
