# Archived: `value_coverage_select`, a placement method that was never a baseline

## What this is

A SoRB-style node-placement rule: walking candidates in order, keep one as a new
node when NO existing node is within the critic's horizon of it, in BOTH
directions. It is SGM's natural ablation partner -- SGM's merge test minus the
two-way consistency check, so the pair isolates what that check buys.

## Why it is archived rather than used

**It was never one of the baselines and it never produced a result.** The
comparison is exactly four methods -- uniform, random, policy (peaks of the
policy's own action entropy), and sgm -- and the user's instruction is that no
new method or approach is added for comparison without their consent.

It was written unprompted while building the harness, sat in
`src/vae_nav/node_placement.py` outside `METHODS`, was never dispatched by any
run, and appears in no result file. It is kept rather than deleted because the
ablation it enables is a real question, and if it is ever wanted it should come
back as a deliberate decision rather than be rewritten from scratch.

## The other reason to keep the record

"policy" was BRIEFLY implemented as this rule before being corrected to entropy
peaks, which is what `node_placement_entropy.py` has always meant by that name.
That correction landed BEFORE the first commit of `node_placement.py` (2b1603e)
and before any evaluation ran, so **no reported number was ever produced with
this definition of `policy`** -- verified against the committed source. This
file exists partly so that claim stays checkable.

## Restoring it

Drop `value_coverage.py` back into `src/vae_nav/`, or paste the function into
`node_placement.py` and add a `coverage` branch to `place()`. It needs only
`H_STEPS` from `vae_nav.value_distance`. Tests for it were removed from
`tests/test_node_placement.py` and are reproduced in `test_value_coverage.py`
here.
