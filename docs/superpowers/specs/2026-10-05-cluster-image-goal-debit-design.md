# Image-goal navigation with DEBiT on the ACG cluster — design

Date: 2026-10-05. Author: the "Trained gated policy" session, for kojo.
Status: **DRAFT FOR REVIEW. Nothing implemented.**

## 1. What this is and why

Train a gated two-mode driving policy whose GOAL is an **image** rather than a
pose, using the DEBiT binocular encoder, in PyTorch, on the ACG cluster.

Three findings drive it:

* **The pose-goal lineage has plateaued on safety, not on reaching.** v43 reaches
  0.92–0.99 at 5–50 m once the action caps came off, but v45 at the full 5–200 m
  band ends at 0.49 with off-road on 0.45 of episodes. Lane crossings and
  off-road, not goal-finding, are what fails.
* **DEBiT reads our imagery.** A frozen DEBiT-B recovers the goal's distance and
  bearing from real CARLA pairs at 5–25 m (distance R² 0.81 on held-out Town03,
  bearing sin 0.92), and is 3–5× more *linearly* decodable than a seg-VAE
  control — which matters because the policy head is shallow. The control
  matches it in-town and loses out of town, so what correspondence buys is
  **generalisation**, which is exactly our weakness.
* **The cluster now works end to end** (CARLA 0.9.16 server in apptainer, client
  and torch 2.5.1 in one conda env, 23 rendered frames on an L40).

### Success criteria

1. A policy that reaches image goals at 5–25 m with **zero fined lane crossings**
   under the standing eval rule (`EVAL_FAULT_ENDS=fault`: lane invasion,
   off-road or collision ends the episode).
2. Measured on **≥100 episodes per town with a stated confidence interval**, on
   two training towns and one **held-out** town.
3. A like-for-like comparison of the two encoder arms (§4), so we can say whether
   DEBiT's extra machinery earns its place.

### Explicit non-goals

Long-range goals (>25 m: a goal image then shares no pixels with the view),
porting the JAX lineage, and beating v43's pose-goal numbers — a different task
is not a fair comparison.

## 2. Task

| | |
|---|---|
| Goal | An RGB frame captured at the goal pose, 5–25 m ahead along the start lane |
| Mode bit | One commanded bit per episode: 1 = image-goal, 0 = explore (coverage) |
| Success | Within the arrival radius of the goal pose (the image is the *cue*, the pose is the *judge*) |
| Lane crossing | **Terminal** in training (kojo's instruction) |
| Episode cap | 1000 steps at 0.1 s |
| Towns | Train Town01 + Town05; hold out Town03 |

**On the terminal crossing rule.** Terminal safety rules have failed three times
here — v15 (0.16), v30, v37 (collapse to 0.01) — each by removing long
trajectories from the buffer so the critic never saw a completed route. All three
ran **without** hindsight relabelling. v42–v45 only worked once HER was fixed.
So "terminal + HER" is genuinely untested rather than a fourth repeat, and §5's
hindsight design is what makes it defensible. **If the goal arm's success is
falling at 150k while episode length shrinks, that is the same failure and we
stop and reconsider rather than running to 1M.**

## 3. Reward

Goal mode, per step, following DEBiT (Bono et al. 2023) with two CARLA-specific
additions:

```
r = -0.01                      slack
  + (d_prev - d_now)           progress, metres of route closed, ON-ROAD ONLY
  - 1.0 * 1[off_road]          shortcut tax (non-terminal)
  + K   on arrival             terminal
crossing of a fined lane marking -> terminal, no bonus
```

* **Progress is counted only while on the road, and off-road costs ≥1.0/step.**
  DEBiT needs no safety term because Habitat's navmesh makes off-mesh
  *physically impossible* and the geodesic is the shortest path on it — no
  shortcut exists to be rewarded. CARLA gives no such guarantee, which is why
  our v16 progress run cut across grass on 31% of steps. The car gains ~0.22 m
  of progress per step at 8 km/h, so any per-step off-road cost above that makes
  a shortcut strictly unprofitable; 1.0 is ~4.5×.
* **K scales with the episode's start distance** (`K = d0`, so arrival is worth
  about as much as the whole route's progress). At their 1.5–5 m scale a fixed
  K=10 is 2× the available progress and dominates; at our 25 m a fixed 10 would
  be a minority of the signal and gives weak reason to finish.
* **Discount 0.99, not 1.0.** The undiscounted return telescopes to (d0 − d_T),
  which is why v16's return looked flat; the discounted critic does not, and at
  0.99 its target is "progress over the next ~100 steps", a local well-shaped
  quantity. γ=1.0 also diverged on the goal critic in v37–v39.
* Explore mode keeps its **own** reward (coverage of new cells, step cost,
  terminal on fault) — one mode bit, two rewards, as in the peer's arm.

## 4. The two arms

Identical except for the perception stack, so the difference is attributable.

**Arm A — frozen DEBiT-B binocular.** `CroCoDownstreamBinocular` (img 112,
patch 16, enc 384×12, dec 512×8, 3136-d output), frozen, fed (observation, goal
image). Output → MLP actor and per-mode critics. No recurrence.

**Arm B — DEBiT's full policy stack** (kojo's choice of second arm). Arm A's
frozen binocular encoder **plus** the trainable monocular ResNet18 branch
**plus** a 1-layer GRU (hidden 512) over [binocular | monocular | prev action],
matching the released DEBiT agent and the peer's recurrent arm. Critics and
actor read the GRU state.

Arm B is the more faithful port and the more expensive; Arm A tells us whether
the recurrence and monocular branch are load-bearing. **Arm A ships first** —
it needs no new sequence machinery and derisks everything else.

## 5. Hindsight relabelling (the part that makes the terminal rule survivable)

Image goals make hindsight unusually natural: **any frame the car actually
reached is a valid goal image for the steps before it.** For an episode of
observations o_1..o_T, relabel with goal = o_j for j > i, recompute the reward
against the pose of o_j, and mark arrival where the car came within the radius.
No re-rendering, no teleport bank — the frames are already in the buffer.

Two rules learned the hard way here: relabelled rows must actually **pay the
arrival bonus** (a min-displacement/radius mismatch made every hindsight row
non-paying in earlier runs), and they must carry the **same** cost structure as
on-policy rows (83% of hindsight rows were mis-costed in v13–v15).

## 6. Algorithm and implementation

Gated TQC in PyTorch, one self-contained learner module with an explicit knob
table that **refuses unknown knobs** and prints a contract line from the live
objects at startup.

| | |
|---|---|
| Critics | Two ensembles (goal, explore), 5 nets × 25 quantiles, drop 2/net in the target |
| Masking | Per-mode, from the **buffer's** episode mode, never from the observation |
| Actor | α_mode·logπ − Q_mode/spread, per-mode temperature, running-std normalisation |
| Discounts | Goal 0.99, explore 0.99 |
| Value clip | Goal critic clipped to the achievable range; checked against the worst realisable return rather than guessed |
| Speed | bf16 autocast, fp32 targets/losses, `torch.compile` **DEFAULT** (`reduce-overhead` stalled the peer's job), GPU-resident fp16 replay, one update per N transitions |
| Env | CARLA 0.9.16 in apptainer, `/workspace/CarlaUE4.sh`, `--nv -e`, **no binds, no Vulkan env** |
| Job | `sbatch`, `PYTHONNOUSERSITE=1`, video/frame dumps saved periodically (cluster runs are headless; kojo's render rule is satisfied by saved video) |

Env count is set by **cores, not VRAM**: CARLA wants ~2–3 cores per server, so
g105 (256 cores / 7 A100) fits far more envs per GPU than g102 (32 / 3 L40).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Terminal crossing collapses the goal arm again | Hindsight (§5); stop rule at 150k if success falls while episode length shrinks |
| DEBiT transfers worse than the probe suggests | The probe measured pose readability, not driving; Arm A is cheap and tells us early |
| Shortcut across non-road | Progress gated on-road + off-road tax above the per-step gain (§3) |
| 0.9.16 vs 0.9.15 | Stated in the paper; cluster arm is a fresh lineage, not a continuation |
| Frozen encoder is the ceiling | If both arms plateau together, the next lever is CroCo-style pretraining on CARLA frames, not more RL |

## 8. Milestones

1. Env wrapper: image goal at 5–25 m, goal frame capture, terminal crossing, DEBiT reward. Tested offline against recorded traces.
2. Learner: gated TQC in torch + replay + hindsight. Unit-tested on a toy env, no CARLA.
3. Smoke run on the cluster: 20k steps, contract line read, no recompiles, throughput measured.
4. Arm A to 1M. Then Arm B.
5. Evaluation: ≥100 episodes/town, fault-ends, two training towns + held-out Town03, CIs stated.

## 9. Open questions for kojo

1. **Arrival radius** for image goals — the pose judge needs one; 2 m matches the
   pose-goal arm.
2. **K = d0, or fixed K?** §3 argues for scaling; fixed is simpler and closer to
   the paper.
3. **Off-road tax of 1.0/step** — calibrated at ~4.5× the per-step gain. Higher
   is safer against shortcuts, but our fines history says bigger numbers have not
   helped before.
4. **Arm order** — Arm A first is my recommendation; say if you want the full
   stack first.
