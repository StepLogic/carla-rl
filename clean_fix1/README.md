# clean_fix1 — pinned shadow tree for the Fix 1 (TQC_VALUE_CEILING) run

WHY THIS EXISTS. As of 2026-09-24 the gated-impl working tree carries another
session's UNCOMMITTED fault-terminating bill regime in
src/vae_nav/gated_reward.py: a fault cap TERMINATES with
-(step_cost + fine) * (1 + remaining) instead of truncating. Any run started
from that tree imports it, so it would NOT be the v10-vel75 baseline and the
ceiling fix could not be attributed.

WHAT IS PINNED. src/ is a copy of the working tree with
src/vae_nav/gated_reward.py restored to committed HEAD (905a244) -- verified:
0 occurrences of "step_cost + _fine", and the counting regime still truncates
at the caps ("v9 COUNTING regime: no terminal bills"). Everything else is the
live code, including the committed Fix 1 / Fix 2 / agent-only-SR changes and
the uncommitted run_gated.sh guards the v10 recipe depends on (without them
GATED_GOAL_MAX_STEPS reverts to 750 and the goal band to 20-50 m).

HOW IT WORKS. Top-level *.py are symlinks to the live files; train_v10.sh,
run_gated.sh and run_gated_her.sh are COPIES (train_v10.sh's hardcoded cd was
repointed here). Every script does sys.path.insert(0, <own dir>/src), so
running them from here imports the pinned src. checkpoints/, results/, logs/,
configs/, dependencies/ and tests/ are symlinks to the live worktree, so
outputs land in the normal places.

NOT A PERMANENT FIXTURE. Once the bill-regime work is committed or set aside,
run from the worktree directly and delete this directory.
