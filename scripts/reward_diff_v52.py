"""What does the running job's reward differ from v52's, term by term?

The arm is supposed to reuse v52's reward -- the one that trains in CARLA. Any
constant the launcher does not export falls back to a CODE DEFAULT, which is not
the same thing as v52's value.
"""
import os
import re
import subprocess

W = "/home/robotlab/scratch/carla-rl/.claude/worktrees/sac-image-goal"
SNAP = (W + "/clean_fix1/results/run_env/"
        "carla_gated_v52_segvae_samesetup_t05.env")

v52 = {}
for line in open(SNAP):
    k, _, v = line.rstrip("\n").partition("=")
    if k:
        v52[k] = v

sb = open(W + "/slurm/train_sac.sbatch").read()
# EVERY assignment on an export line, not just the first: lines like
# `export A=1 B=2 C=3` are common here and a first-match regex silently hides
# B and C, which reads as drift that is not there.
exported = {}
for line in sb.splitlines():
    line = line.strip()
    if not line.startswith("export "):
        continue
    for k, v in re.findall(r"([A-Z0-9_]+)=([^\s]+)", line[len("export "):]):
        exported.setdefault(k, v)

# Every env var the two reward modules read.
src = ""
for f in ("src/vae_nav/gated_reward.py", "src/vae_nav/lane_explore_reward.py",
          "src/vae_nav/lane_follow_reward.py"):
    p = os.path.join(W, f)
    if os.path.exists(p):
        src += open(p).read()
names = sorted(set(re.findall(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"\s*,\s*([^)]+)\)', src)))

print(f"{'VARIABLE':38s} {'v52':>10s} {'THIS RUN':>12s}   source")
print("-" * 86)
drift = []
for name, default in names:
    default = default.strip().strip('"').strip("'")
    mine = exported.get(name)
    src_lbl = "exported" if mine is not None else f"DEFAULT({default})"
    if mine is None:
        mine = default
    theirs = v52.get(name)
    if theirs is None:
        continue                      # not in v52's snapshot: no comparison
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    a_, b_ = _num(theirs), _num(mine)
    # Compare NUMERICALLY where both parse: "1" and "1.0" are the same setting,
    # and a string diff would bury the real drift in formatting noise.
    same = (a_ == b_) if (a_ is not None and b_ is not None) else \
        str(theirs).strip() == str(mine).strip()
    if not same:
        drift.append((name, theirs, mine, src_lbl))
    print(f"{name:38s} {theirs:>10s} {mine:>12s}   {src_lbl}"
          f"{'' if same else '   <-- DRIFT'}")

print()
if drift:
    print(f"{len(drift)} TERM(S) DIFFER FROM v52:")
    for n, t, m, s in drift:
        print(f"  {n}: v52={t}  this run={m}  ({s})")
else:
    print("no drift: every term matches v52")
