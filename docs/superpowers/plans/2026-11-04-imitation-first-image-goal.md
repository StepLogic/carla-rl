# Imitation-First Image-Goal Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the image-goal SAC agent to drive by cloning CARLA's autopilot,
then fine-tune it with v30's reward and the user's 200 m goal chaining.

**Architecture:**
- **M0.** Vendor CARLA 0.9.16's `agents` package so `BasicAgentExpert` runs
  on the cluster, then benchmark it as a hard gate.
- **M1.** A recording mode in `train_carla_sac.py` drives the expert (later
  the policy, for DAgger) through the unchanged env stack and writes one
  `.npz` per episode.
- **M2.** A behaviour-cloning mode trains the existing `GatedRecurrentSAC`
  network on those shards. The checkpoint format is unchanged, so `run_eval`,
  `run_trace` and `INIT_FROM` all work as before.
- **M3.** Measurement, with two eval loopholes closed.
- **M4.** DAgger, only if M3 says so.
- **M5.** SAC fine-tune with a frozen-encoder critic warm-up and a decaying
  cloning loss, only if M3/M4 clear the bar.

**Tech Stack:** Python 3.10, PyTorch 2.5, CARLA 0.9.16 (server in
`carla0916nv.sif`, client in `~/carla-cluster/envs/carla310`), numpy, Slurm on
the ACG cluster.

**Spec:** `docs/superpowers/specs/2026-11-04-imitation-first-image-goal-design.md`

## Global Constraints

- Operating speed: expert target **15 km/h**; final approach stays at
  `BasicAgentExpert.FINAL_APPROACH_KMH` (6 km/h).
- Training towns: **Town01, Town04, Town05, Town10HD**.
- Held out: **Town02, Town03**. Never recorded, never trained on, never used
  for checkpoint choice or tuning.
- Action space unchanged: steer [−1, 1], throttle [0, 1], no brake, no
  reverse. `MIN_THROTTLE` stays unset (0).
- Actor observation unchanged: frame, mode bit, binocular vector.
- Evals:
  - Faults (crossing, off-road, collision) end the episode.
  - Arrival is within 2 m at ≤10 km/h.
  - Headline numbers use n=100 paired true/wrong goal image.
- Every new overlay exports `MAX_DISTANCE_M=1000000000`. Eval overlays (and
  only eval overlays) also export `GOAL_KEEP_ON_PASS=1`.
- Runs source `slurm/v52_reward.env`, plus a named overlay via
  `REWARD_OVERLAY`. Launch settings live in the overlay, never on the
  command line.
- All training and evals run on the cluster.
  - Outputs go under `checkpoints/` or `logs/`, because
    `slurm/sync_to_cluster.sh` uses `--delete`.
  - Never sync while a job is starting up: the rlib package is briefly empty
    mid-sync.
- M0 gate: arrivals **≥ 0.95** on 5-25 m goals, AND **≥ 1000 m** mean between
  failures over **≥ 2000 m** of explore driving.
- Recovery noise: per episode σ ∈ {0, 0.1, 0.2}, on steering only. The stored
  label is always the expert's clean action.
- Explore-mode branch rule: the smallest heading change; where no exit is
  within 30° of straight, turn right (positive Δyaw in CARLA's left-handed
  frame).
- DAgger:
  - Replan the expert's route when the car is **> 3 m** from it.
  - **At most 2 rounds**; if still below the bar, stop and report to the user
    before M5.
- Success bar:
  - **≥ 300 m** mean between failures on Town02 and Town03.
  - Town02 SR **≥ 0.80** with wrong-goal SR **≤ 0.10** (McNemar p < 0.001).
  - Town03 SR **≥ 0.60**.
  - **≥ 50%** of 200 m chains completed.
  - **2 seeds** for the final recipe.
- Tests run with:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES= /usr/bin/python3 -m pytest -q <files> -p no:cacheprovider`.
  The full suite has 44 known environment failures; a task must not add any.

## Review Focus

1. **The expert has no plan at episode start, or loses it mid-episode.**
   `act()` returns None. No plan at step 0: discard the episode. Losing it
   later: end the episode as truncated and keep the rows recorded so far.
   (Task 4, `test_keep_episode_*`)
2. **A fault on the first step, giving a zero-length episode.** It is never
   saved, and loading code skips it without crashing.
   (Task 3, `test_fill_buffer_skips_empty_episodes`)
3. **An explore route that dead-ends before its length.** It ends truncated,
   and the gate does not count that as a failure.
   (Task 2, `test_dead_end_returns_partial_route`;
   Task 4, `test_gate_counts_only_fault_ends`)
4. **A frame at the env's 120 px instead of the row's 128 px.** It is stored
   only after `pxr.to_frame(..., MONO_IMG)`; `save_episode` rejects anything
   else. (Task 3, `test_save_rejects_wrong_frame_shape`)
5. **A recording or cloning job pointed at Town02 or Town03 by mistake.** It
   refuses to run. (Task 4, `test_held_out_towns_refused`)

---

## Milestone M0 — expert on the cluster, and the gate

### Task 1: Vendor CARLA 0.9.16 `agents` and import it from the repo

**Files:**
- Create: `dependencies/carla_agents/agents/**` (copied unmodified from the container)
- Modify: `src/vae_nav/expert_oracle.py` (the two hard-coded `"/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla"` paths)
- Test: `tests/test_carla_agents_path.py`

**Interfaces:**
- Produces: `expert_oracle.carla_agents_path() -> str`. Returns
  `os.environ["CARLA_AGENTS_PATH"]` if set, else
  `<repo>/dependencies/carla_agents`. Every `agents` import in
  `expert_oracle.py` goes through it.

- [ ] **Step 1: Copy the package out of the container (compute node; `apptainer exec` fails on the login node)**

```bash
ssh login1.acg.maine.edu 'srun --partition=gpu --time=0:05:00 --cpus-per-task=1 --mem=4gb bash -lc \
  "module load apptainer; apptainer exec $HOME/carla-cluster/carla0916nv.sif \
   tar -C /workspace/PythonAPI/carla -cf - agents" > ~/carla_agents_0916.tar'
scp login1.acg.maine.edu:carla_agents_0916.tar /home/robotlab/.claude/jobs/c1368d5c/tmp/
mkdir -p dependencies/carla_agents && tar -C dependencies/carla_agents -xf /home/robotlab/.claude/jobs/c1368d5c/tmp/carla_agents_0916.tar
ls dependencies/carla_agents/agents/navigation/basic_agent.py
```
Expected: the file exists.

- [ ] **Step 2: Write the failing tests**

```python
def test_agents_path_defaults_to_vendored_copy(monkeypatch):
    monkeypatch.delenv("CARLA_AGENTS_PATH", raising=False)
    p = expert_oracle.carla_agents_path()
    assert p.endswith(os.path.join("dependencies", "carla_agents"))
    assert os.path.isfile(os.path.join(p, "agents", "navigation", "basic_agent.py"))

def test_agents_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CARLA_AGENTS_PATH", str(tmp_path))
    assert expert_oracle.carla_agents_path() == str(tmp_path)

def test_no_hard_coded_local_carla_path_remains():
    src = open(expert_oracle.__file__).read()
    assert "/home/robotlab/Apps/CARLA_0.9.15" not in src
```

- [ ] **Step 3: Run them; expect `AttributeError: carla_agents_path`**

- [ ] **Step 4: Implement `carla_agents_path()` and route both imports through it** (prepend it to `sys.path` once).

- [ ] **Step 5: Run the tests; expect PASS**

- [ ] **Step 6: Prove the import works on the cluster**

`bash slurm/sync_to_cluster.sh`, then:
```bash
ssh login1.acg.maine.edu 'cd ~/carla-rl && PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:$PWD/src ~/carla-cluster/envs/carla310/bin/python -c \
 "import sys; from vae_nav.expert_oracle import carla_agents_path; sys.path.insert(0, carla_agents_path()); import agents.navigation.basic_agent as b; print(b.__file__)"'
```
Expected: a path under `~/carla-rl/dependencies/carla_agents`.

- [ ] **Step 7: Commit**
`git add dependencies/carla_agents src/vae_nav/expert_oracle.py tests/test_carla_agents_path.py && git commit -m "Vendor CARLA 0.9.16 agents; expert imports it from the repo"`

### Task 2: Explore route rule and expert plan hooks

**Files:**
- Create: `src/vae_nav/expert_route.py`
- Modify: `src/vae_nav/expert_oracle.py` (`BasicAgentExpert`)
- Test: `tests/test_expert_route.py` (fake waypoint graph with `.next(d)`, `.is_junction`, `.transform.location`, `.transform.rotation.yaw`; reuse the pattern in `tests/test_goal_chain.py`)

**Interfaces:**
- Produces:
  - `explore_route(start_wp, length_m: float, step_m: float = 2.0, straight_tol_deg: float = 30.0) -> list`
    - Returns waypoints from `start_wp`, inclusive.
    - At a fork, takes the successor with the smallest `|Δyaw|`. If none is
      within `straight_tol_deg`, takes the right-turning one (Δyaw > 0) with
      the smallest `|Δyaw|`, falling back to the smallest `|Δyaw|` overall.
    - `Δyaw` is wrapped to (−180, 180].
    - Stops once `length_m` is covered, but never inside a junction (walks
      out of one, up to +40 m). Returns what it has at a dead end.
  - `needs_replan(route_xy: np.ndarray, xy, max_off_m: float) -> bool`.
    True when the nearest route point is farther than `max_off_m`; also True
    for an empty route.
  - `BasicAgentExpert.set_explore_plan(waypoints: list) -> int`.
    - Sets a global plan of `(wp, RoadOption.LANEFOLLOW)` with `clean_queue=True`.
    - Clears the PID buffers as `reset()` does.
    - Sets `_goal_loc = None`, so there is no final approach.
    - Returns the queue length.
  - `BasicAgentExpert.act()`. Returns None once an explore plan is done.
  - `BasicAgentExpert.replan_if_off_route(max_off_m: float = 3.0) -> bool`.
    - Uses `needs_replan` on the local planner's remaining queue.
    - On a replan: goal mode re-calls `reset(self._goal_loc)`; explore mode
      calls `set_explore_plan(explore_route(<current wp>, self._explore_remaining_m))`.
    - Returns whether it replanned.

- [ ] **Step 1: Write the failing tests**

```python
def test_straight_exit_preferred(): ...      # fork: 0° and +90° -> route continues at 0°
def test_t_junction_turns_right(): ...       # fork: -90° and +90° -> route takes +90°
def test_left_only_fork_takes_the_smallest_turn(): ...   # -90° and -150° -> -90°
def test_route_length_is_respected(): ...    # 100 m request on a long road -> covered length in [100, 102]
def test_route_never_ends_inside_a_junction(): ...       # last waypoint .is_junction is False
def test_dead_end_returns_partial_route(): ...           # 20 m road, 100 m request -> 11 waypoints, no error
def test_needs_replan():
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    assert not needs_replan(xy, (2.0, 2.9), 3.0)
    assert needs_replan(xy, (2.0, 3.1), 3.0)
    assert needs_replan(np.zeros((0, 2)), (0.0, 0.0), 3.0)
```

- [ ] **Step 2: Run them; expect `ModuleNotFoundError: vae_nav.expert_route`**

- [ ] **Step 3: Implement `explore_route` and `needs_replan` in `src/vae_nav/expert_route.py`**, plus the three `BasicAgentExpert` methods. The `BasicAgentExpert` methods are thin CARLA glue and are proven by the Task 5 cluster smoke.

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Commit**
`git commit -m "Explore route branch rule; expert explore plans and off-route replanning"`

## Milestone M1 — recording

### Task 3: Episode shard format and buffer loading

**Files:**
- Create: `torch_rl/expert_data.py`
- Test: `tests/test_expert_data.py`

**Interfaces:**
- Produces:
  - One `.npz` per episode at `<root>/<town>/ep_<seed>.npz`, with keys:
    - Per step: `frames` (T+1, 128, 128, 3) uint8; `binoc` (T+1, 3136)
      float16; `act_exec` (T, 2) float32, actor space; `act_label` (T, 2)
      float32, actor space; `reward` (T,) float32; `speed` (T,) float32;
      `pos` (T, 2) float32.
    - Per episode: `mode` int8; `terminal` bool; `truncated` bool; `goal` =
      `pxr.goal_block(goal_frame)`; `sigma` float32; `driver` str; `end` str;
      `town` str; `seed` int64.
  - `save_episode(path: str, ep: dict) -> None`. Validates shapes and dtypes,
    raises `ValueError` on any mismatch, refuses T == 0.
  - `load_episode(path: str) -> dict`.
  - `episode_paths(root: str, towns: Sequence[str]) -> list[str]` (sorted).
  - `split_by_episode(paths: list[str], val_frac: float = 0.05, seed: int = 0) -> tuple[list[str], list[str]]`
    (deterministic; at least one validation episode when `len(paths) >= 2`).
  - `fill_buffer(buf: DeviceEpisodeBuffer, paths: list[str], which: str = "label", env_id: int = 0) -> int`.
    - Each row is rebuilt as
      `pxr.mono_row(frames[t], np.array([mode], np.float32), binoc[t].astype(np.float32))`.
    - The action is `act_label` (`which="label"`) or `act_exec` (`"exec"`).
    - Terminal and truncated flags are set only on the last step.
    - Calls `buf.end_episode(env_id)` after each episode, skips T == 0, and
      returns the number of transitions added.

- [ ] **Step 1: Write the failing tests**

```python
def test_round_trip(tmp_path): ...  # save then load -> identical arrays, scalars and strings
def test_save_rejects_wrong_frame_shape(tmp_path):
    ep = _ep(T=3); ep["frames"] = np.zeros((4, 120, 120, 3), np.uint8)
    with pytest.raises(ValueError, match="frames"): save_episode(str(tmp_path / "e.npz"), ep)
def test_save_rejects_empty_episode(tmp_path): ...  # T == 0 -> ValueError
def test_fill_buffer_uses_the_requested_actions(tmp_path): ...
    # act_label all +0.5, act_exec all -0.5 -> sampled chunk actions are +0.5 for "label", -0.5 for "exec"
def test_fill_buffer_skips_empty_episodes(tmp_path): ...  # a T == 0 npz written directly is skipped, count excludes it
def test_split_is_deterministic_and_by_episode(): ...
    # same seed -> same split; no path in both; val non-empty for >= 2 paths
```
(Tests use a CPU `DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 400, "cpu", goal_dim=pxr.PIX)`.)

- [ ] **Step 2: Run them; expect `ModuleNotFoundError: torch_rl.expert_data`**
- [ ] **Step 3: Implement the five functions in `torch_rl/expert_data.py`**
- [ ] **Step 4: Run them; expect PASS**
- [ ] **Step 5: Commit** `git commit -m "Expert episode shards: format, validation, buffer loading"`

### Task 4: Pure recording logic

**Files:**
- Create: `torch_rl/expert_record.py` (pure parts only in this task)
- Test: `tests/test_expert_record.py`

**Interfaces:**
- Produces:
  - `HELD_OUT_TOWNS = ("Town02", "Town03")`.
  - `assert_recordable_town(town: str) -> None` (raises `ValueError` for held-out towns).
  - `episode_plan(seed: int) -> dict`. Returns `{"mode": 1 if seed % 2 == 0 else 0, "sigma": (0.0, 0.1, 0.2)[(seed // 2) % 3]}`.
  - `dart_action(a_env: np.ndarray, sigma: float, rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> np.ndarray`.
    Adds N(0, σ) to index 0 only and clips to [low, high]. At σ = 0 it is a copy.
  - `record_step(driver: str, a_policy, a_expert: np.ndarray, sigma: float, rng, low, high) -> tuple[np.ndarray, np.ndarray]`.
    Returns `(a_exec, a_label)`. Driver `"expert"` gives
    `(dart_action(a_expert, ...), a_expert)`; driver `"policy"` gives
    `(a_policy, a_expert)`.
  - `keep_episode(n_steps: int) -> bool`. Returns `n_steps > 0`.
  - `gate_report(goal_eps: list[dict], explore_eps: list[dict], min_arrival: float = 0.95, min_m_between: float = 1000.0, min_explore_m: float = 2000.0) -> dict`.
    - Episode dicts carry `end` and `path_m`.
    - Faults are `end in {"crossing", "off_road", "collision"}`.
    - Returns `{"arrival_rate", "explore_m", "faults", "m_between" (inf if 0 faults), "pass": bool, "reason": str}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_held_out_towns_refused():
    for t in ("Town02", "Town03"):
        with pytest.raises(ValueError, match="held out"): assert_recordable_town(t)
    assert_recordable_town("Town05")
def test_episode_plan_is_balanced():
    plans = [episode_plan(s) for s in range(600)]
    assert sum(p["mode"] for p in plans) == 300
    assert {p["sigma"] for p in plans} == {0.0, 0.1, 0.2}
def test_dart_noise_touches_steer_only_and_clips(): ...
def test_dart_zero_sigma_is_identity(): ...
def test_record_step_labels_are_always_the_expert():
    a_exec, a_label = record_step("policy", np.array([0.3, 0.4]), np.array([-0.2, 0.6]), 0.2, rng, low, high)
    assert np.allclose(a_exec, [0.3, 0.4]) and np.allclose(a_label, [-0.2, 0.6])
def test_keep_episode_drops_only_empty(): ...
def test_gate_counts_only_fault_ends():
    rep = gate_report([{"end": "reached", "path_m": 20}] * 20,
                      [{"end": "cap", "path_m": 150}] * 14 + [{"end": "crossing", "path_m": 100}])
    assert rep["faults"] == 1 and rep["m_between"] == pytest.approx(2200.0) and rep["pass"]
def test_gate_fails_on_low_arrivals(): ...      # 18/20 reached -> pass False, reason mentions "arrival"
def test_gate_needs_enough_explore_driving(): ...   # 0 faults over 900 m -> pass False, reason "insufficient"
```

- [ ] **Step 2: Run them; expect import errors**
- [ ] **Step 3: Implement the pure functions**
- [ ] **Step 4: Run them; expect PASS**
- [ ] **Step 5: Commit** `git commit -m "Recording logic: plan schedule, recovery noise, gate report"`

### Task 5: Recording loop, mode dispatch, M0 benchmark (GATE), M1 recording

**Files:**
- Modify: `torch_rl/expert_record.py` (add `run_record`)
- Modify: `train_carla_sac.py`. Extract `build_agent`, add the settings, dispatch.
- Create: `slurm/il_expert.env`, `slurm/il_record.sh`

**Interfaces:**
- Consumes:
  - Task 2: `BasicAgentExpert` with `set_explore_plan` and `replan_if_off_route`; `explore_route`.
  - Task 3: `save_episode`.
  - Task 4: `episode_plan`, `record_step`, `keep_episode`, `gate_report`, `assert_recordable_town`.
- Produces:
  - `train_carla_sac.build_agent(binoc_dim: int, act_dim: int) -> GatedRecurrentSAC`.
    - The config and construction are moved verbatim out of `main()`,
      including `load_debit_mono` and the `compress_targ` sync.
    - `main()` calls it.
  - Settings:
    - `EXPERT_RECORD` (bool); `EXPERT_DRIVER` (`expert` | `policy`); `EXPERT_RECORD_DRY` (bool).
    - `EXPERT_RECORD_EPISODES` (int, default 200); `EXPERT_SEED0` (int, default 100000).
    - `EXPERT_TARGET_KMH` (default 15.0); `EXPERT_EXPLORE_MIN_M` / `EXPERT_EXPLORE_MAX_M` (100 / 200); `EXPERT_REPLAN_M` (3.0).
    - `EXPERT_RECORD_ROOT` (default `$HOME/carla-rl/checkpoints/il_data`).
  - `run_record(env, agent, pxr, server, out_root) -> None`.
    - Writes `<root>/<town>/ep_<seed>.npz` (unless dry) and `<out_root>/record_summary.json`.
      The summary contains the per-episode list `{seed, mode, sigma, end, path_m, steps}`,
      that town's `gate_report`, and step totals.
    - Prints `[RecordDone] {json}`.
  - Dispatch in `main()`, placed BEFORE the `if EVAL_FROM:` block:
    `if EXPERT_RECORD: (load EVAL_FROM when driver == policy) return run_record(...)`.
  - The env is built with `eval_mode=bool(EVAL_FROM) or EXPERT_RECORD`.

**Loop rules:**
- **Per episode `k`:**
  - `seed = EXPERT_SEED0 + k`.
  - Set the `GatedReward` wrapper's `force_mode` from `episode_plan(seed)` before `env.reset(seed=seed)`.
  - Seed `random`, `np.random` and `torch` as `run_eval` does.
  - Build the expert as `BasicAgentExpert(core.hero, core.map, env.action_space, target_speed_kmh=EXPERT_TARGET_KMH)`,
    and rebuild it whenever `core.hero.id` changes.
- **Planning:**
  - Goal mode: `expert.reset(info["goal_location"])`.
  - Explore mode: `expert.set_explore_plan(explore_route(map.get_waypoint(hero location), U[MIN, MAX]))`.
- **Each step:**
  - Policy driver only: call `replan_if_off_route(EXPERT_REPLAN_M)` first.
  - `a_expert = expert.act()`. None at step 0: discard the episode. None later: end it as truncated.
  - Store `pxr.to_frame(info["debit_cur"], pxr.MONO_IMG).astype(np.uint8)`, `obs["binoc"]` as fp16, and `to_actor(a_exec)` / `to_actor(a_label)`.
  - Then step the env with `a_exec`.
- **Episode end:**
  - The first `_fault(info)`, the env's `term` or `trunc`, or (explore) the expert's plan being done (truncated).

- [ ] **Step 1: Extract `build_agent`; run the full suite; expect 44 failed, the rest passed, no new failures**
- [ ] **Step 2: Implement `run_record` and the dispatch; run the full suite again; expect no new failures**

- [ ] **Step 3: Write `slurm/il_expert.env`**

```bash
. "$REPO/slurm/v30_overlay.env"
export MAX_DISTANCE_M=1000000000
export GOAL_CAM_MATCH=1 GOAL_DIST_MIN=5 GOAL_DIST_MAX=25 JUNCTION_GOAL_FRAC=0.85
export EXPERT_TARGET_KMH=15 EXPERT_EXPLORE_MIN_M=100 EXPERT_EXPLORE_MAX_M=200 EXPERT_REPLAN_M=3
```

- [ ] **Step 4: Write `slurm/il_record.sh`**
  - Loops `Town01 Town04 Town05 Town10HD`, with `EXPERT_SEED0=$((100000 + 10000 * i))`.
  - Each iteration runs `REWARD_OVERLAY=slurm/il_expert.env ENV_TOWNS=$town EXPERT_RECORD=1 RUN_NAME=il_record_$town CARLA_PORT=34$i0 bash slurm/train_sac.sbatch`.
  - `EXPERT_RECORD_DRY` and `EXPERT_RECORD_EPISODES` pass through from the caller.

- [ ] **Step 5: Cluster smoke (2 episodes, dry, Town05)**
  - Sync.
  - Run `EXPERT_RECORD_DRY=1 EXPERT_RECORD_EPISODES=2` for Town05 only, via `sbatch --wrap`.
  - Expected: one goal and one explore episode, `[RecordDone]` printed, no Traceback.

- [ ] **Step 6: M0 benchmark**
  - `EXPERT_RECORD_DRY=1 EXPERT_RECORD_EPISODES=80 bash slurm/il_record.sh` on the spare L40.
  - Prove it alive (process, fresh log, GPU).
  - Expected: four `record_summary.json` files.

- [ ] **Step 7: GATE**
  - Combine the four towns' `gate_report` inputs and run `gate_report`.
  - **If `pass` is false: stop here and report the numbers to the user** (a
    brake decision is theirs).
  - Otherwise continue.

- [ ] **Step 8: M1 recording**
  - `EXPERT_RECORD_EPISODES=200 bash slurm/il_record.sh` (not dry).
  - Expected: about 300k steps total across `record_summary.json`. Top up any
    town short of 60k steps with a second seed range.

- [ ] **Step 9: Commit**
`git commit -m "Expert recording mode; build_agent extracted; il_expert overlay"`

## Milestone M2 — behaviour cloning

### Task 6: Cloning loss and training mode

**Files:**
- Create: `torch_rl/bc_trainer.py`
- Modify: `train_carla_sac.py` (`BC_TRAIN` dispatch at the top of `main()`, before any server starts)
- Test: `tests/test_bc_trainer.py` (CPU agent from `tests.test_one_critic._cfg` / `_chunk`)

**Interfaces:**
- Consumes: Task 3 `episode_paths`, `split_by_episode`, `fill_buffer`; Task 5 `build_agent`.
- Produces:
  - `bc_loss(agent, chunk: dict, clip: float = 0.995) -> tuple[torch.Tensor, dict]`. The stats dict has keys `bc_nll`, `bc_mae`, `bc_log_std`.
  - `make_bc_optimizer(agent, lr: float = 3e-4) -> torch.optim.Adam` over `compress`, `act_embed` (if not None), `core` and `actor`.
  - `sync_targets(agent) -> None`. `*_targ.load_state_dict(online.state_dict())` for compress, act_embed and core.
  - `bc_step(agent, opt, chunk: dict, weight: float = 1.0) -> dict`.
  - `run_bc_train(out_root: str) -> None`, driven by these settings:
    - `BC_DATA_ROOT`; `BC_UPDATES` (default 20000); `BC_CHUNKS` (32); `BC_LEN` (40).
    - `BC_SAVE_EVERY` (2500); `BC_INIT_FROM` (optional checkpoint, used for DAgger rounds).
  - Checkpoints go to `<out_root>/step_<n>/` via `agent.save`, after `sync_targets`.

The loss is the pre-tanh Gaussian NLL, on the same alignment as `_update_impl`:
```python
K = agent.cfg.burn_in
s_all = agent._roll(agent.compress, agent.act_embed, agent.core, obs, keep, goal)
s  = s_all[:, :-1].reshape(N, H)                       # grads flow into compress/core
st = agent._state(obs[:, K:-1]).reshape(N, state_dim)
mean, log_std = agent._actor_forward(torch.cat([s, st], -1))
u = torch.atanh(actions.reshape(N, A).clamp(-clip, clip))
nll = (0.5 * ((u - mean) / log_std.exp()) ** 2 + log_std).sum(-1)
loss = (nll * valid.reshape(N)).sum() / valid.sum().clamp_min(1)
```
(`goal` is `None` for this config; the chunk keys are `_CHUNK_KEYS`.)

- [ ] **Step 1: Write the failing tests**

```python
def test_bc_loss_decreases_on_a_fixed_chunk():  # 30 bc_step calls -> final loss < 0.5 * first
def test_bc_gradients_reach_encoder_core_and_actor():
    loss, _ = bc_loss(agent, chunk); loss.backward()
    for mod in (agent.compress, agent.core, agent.actor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
def test_bc_loss_ignores_invalid_rows():  # changing actions where valid == 0 leaves the loss unchanged
def test_sync_targets_copies_online_weights():
def test_bc_checkpoint_reloads(tmp_path):  # save -> load into a fresh agent -> identical deterministic act()
```

- [ ] **Step 2: Run them; expect import errors**
- [ ] **Step 3: Implement `bc_trainer.py` and the `BC_TRAIN` dispatch.**
  `run_bc_train` fills a `DeviceEpisodeBuffer` with `store_device=REPLAY_DEVICE` and `u8_cols=pxr.MONO_PIX`, using `which="label"`. It logs `[BC] step= train_nll= val_nll= mae=` every 250 updates; validation is 8 chunks from the validation split.
- [ ] **Step 4: Run the tests; expect PASS. Run the full suite; expect no new failures**
- [ ] **Step 5: Commit** `git commit -m "Behaviour cloning: pre-tanh NLL over encoder, GRU and actor"`

### Task 7: Cloning run and checkpoint choice on training-town validation

**Files:**
- Create: `slurm/il_bc.env`, `slurm/il_eval_val.env`, `scripts/select_checkpoint.py`
- Test: `tests/test_select_checkpoint.py`

**Interfaces:**
- Produces: `select_checkpoint(results: dict[int, tuple[float, float]], max_wrong: float = 0.10) -> int`.
  - The input maps step to `(true_sr, wrong_sr)`, pooled over the validation towns.
  - Returns the step with the highest `true_sr` among those with
    `wrong_sr <= max_wrong`, ties going to the later step.
  - If no step qualifies, returns the step with the largest
    `true_sr - wrong_sr`.

- [ ] **Step 1: Write the failing tests**

```python
def test_highest_true_among_qualifying():
    assert select_checkpoint({2500: (0.6, 0.05), 5000: (0.7, 0.2), 7500: (0.65, 0.1)}) == 7500
def test_ties_go_to_the_later_step(): ...
def test_falls_back_to_the_largest_gap_when_none_qualify(): ...
```
- [ ] **Step 2: Run them; expect failures**

- [ ] **Step 3: Implement `select_checkpoint` plus its CLI** (eval directories in, chosen step out)

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Write `slurm/il_bc.env`**

```bash
. "$REPO/slurm/il_expert.env"
export BC_TRAIN=1 BC_DATA_ROOT=$HOME/carla-rl/checkpoints/il_data BC_UPDATES=20000 BC_CHUNKS=32 BC_LEN=40
export REPLAY_DEVICE=cpu REPLAY_U8=1
```

- [ ] **Step 6: Write `slurm/il_eval_val.env`**

```bash
. "$REPO/slurm/il_expert.env"
export GOAL_KEEP_ON_PASS=1
```

- [ ] **Step 7: Launch the cloning run and prove it alive**
  - Command: `REWARD_OVERLAY=slurm/il_bc.env RUN_NAME=il_bc_r0 sbatch ... slurm/train_sac.sbatch` (an L40, 12 CPUs, 160 GB memory).
  - Expected: `[BC]` lines with `val_nll` falling; checkpoints at `step_2500 ... step_20000`.

- [ ] **Step 8: Evaluate every checkpoint on the validation set**
  - Towns: Town05 and Town01. Seeds: `EVAL_SEED=50000`, 40 episodes each. Both true and `WRONG_GOAL_MODE=1`, under `il_eval_val.env`.
  - Pool the results and run `select_checkpoint`.
  - Expected: one chosen step, written to `logs/il_bc_r0_selected.txt`.

- [ ] **Step 9: Commit** `git commit -m "Checkpoint choice on training-town validation; cloning overlays"`

## Milestone M3 — measurement (decision point)

### Task 8: Eval fixes, v30 re-score, clone measurement

**Files:**
- Create: `slurm/il_eval.env`, `slurm/v30_eval_fixed.env`, `scripts/trace_failure_stats.py`
- Test: `tests/test_trace_failure_stats.py`

**Interfaces:**
- Produces: `failure_stats(segments: list[dict]) -> dict`.
  - Each segment has `metres` and `end`.
  - Returns `{"metres", "faults", "mean_between", "median_run", "longest"}`.
  - Faults are `end in {"crossing", "off_road", "collision"}`.
  - `mean_between = metres / faults`, or inf when there are no faults.
  - `median_run` and `longest` are taken over fault-ended segments.
  - The CLI takes trace directories and prints one row each.

- [ ] **Step 1: Write the failing tests**

```python
def test_mean_between_failures():
    s = [{"metres": 30, "end": "crossing"}, {"metres": 50, "end": "cap"}, {"metres": 20, "end": "off_road"}]
    r = failure_stats(s); assert r["faults"] == 2 and r["mean_between"] == pytest.approx(50.0)
def test_no_faults_is_infinite(): ...
def test_matches_the_published_v30_town02_figure():  # a fixture of trace_v30det_Town02's index gives 31.7 m
```
- [ ] **Step 2: Run them; expect failures**

- [ ] **Step 3: Implement `failure_stats` and its CLI**

- [ ] **Step 4: Run the tests; expect PASS**

- [ ] **Step 5: Write the two eval overlays**
  - `il_eval.env`: `. il_expert.env` plus `export GOAL_KEEP_ON_PASS=1`.
  - `v30_eval_fixed.env`: `. v30_eval.env` plus `export MAX_DISTANCE_M=1000000000 GOAL_KEEP_ON_PASS=1`.

- [ ] **Step 6: Re-score v30 under the fixed rules**
  - `checkpoints/sac_debitmono_t05t01_v30/step_300000`, Town02 and Town03, n=100, true and wrong, `v30_eval_fixed.env`.
  - Expected: two eval.json pairs. Record them next to the published 0.560 / 0.030 and 0.350 / 0.040.

- [ ] **Step 7: Measure the selected clone**
  - Navigation: Town02 and Town03, n=100, true and wrong, `il_eval.env`.
  - Lane keeping: explore traces with `EVAL_TRACE=1 EVAL_FORCE_MODE=0 TRACE_NOISE_SCALE=0 TRACE_END_ON_CROSSING=1 TRACE_DIST_M=5000`.
  - Run `trace_failure_stats.py` on the traces and `~/paired_wrong.py` on each true/wrong pair.
  - Report mean speed (`path_m / steps / 0.05 * 3.6` km/h) beside every lane-keeping figure, for the clone and the v30 re-score alike.

- [ ] **Step 8: DECISION**
  - Mean between failures **≥ 300 m on both towns:** go to Task 10.
  - Otherwise: go to Task 9.
  - Either way, report the numbers to the user first.

- [ ] **Step 9: Commit** `git commit -m "Eval fixes (200 m cap, moved goal); failure stats; v30 re-scored"`

## Milestone M4 — DAgger (only if Task 8 sends it here)

### Task 9: Policy-driven relabelling rounds

**Files:**
- Create: `slurm/il_dagger.sh`, `slurm/il_bc_dagger.env`

**Interfaces:**
- Consumes: Task 5 `run_record` with `EXPERT_DRIVER=policy` (Task 4 `record_step` already pins exec = policy and label = expert), and Task 6 `BC_INIT_FROM`.

- [ ] **Step 1: Write `slurm/il_dagger.sh`.** It is `il_record.sh` with:
  - `EXPERT_DRIVER=policy`;
  - `EVAL_FROM=<selected clone>`;
  - `EXPERT_SEED0=$((200000 + 10000 * i + 100000 * round))`;
  - `EXPERT_RECORD_ROOT=$HOME/carla-rl/checkpoints/il_data` (the data is aggregated with M1's);
  - `EXPERT_RECORD_EPISODES=120`.
- [ ] **Step 2: Write `slurm/il_bc_dagger.env`:** `. il_bc.env` plus `export BC_INIT_FROM=<previous selected> BC_UPDATES=10000`.
- [ ] **Step 3: Round 1.** Record, retrain, then repeat Task 7 steps 7-8 and Task 8 step 7. Expected: the mean between failures rises over the clone's.
- [ ] **Step 4: Round 2 if still below 300 m.** If it is still below after round 2, **stop and report to the user before M5.**
- [ ] **Step 5: Commit** `git commit -m "DAgger rounds: policy-driven recording with expert relabels"`

## Milestone M5 — reinforcement fine-tune (only once lane keeping clears the bar)

### Task 10: Critic warm-up with a frozen encoder, and a decaying cloning loss

**Files:**
- Modify: `torch_rl/gated_recurrent_tqc.py`, `torch_rl/bc_trainer.py`, `train_carla_sac.py`
- Test: `tests/test_il_finetune.py`

**Interfaces:**
- Consumes: Task 3 `fill_buffer`; Task 6 `bc_step` and `make_bc_optimizer`.
- Produces:
  - `GatedRecurrentTQC.set_warmup(on: bool) -> None`. Sets `requires_grad=not on` on compress, act_embed and core. While on, `_update_impl` skips the actor and temperature section and returns `actor_loss = 0`.
  - `bc_trainer.bc_weight(n_updates: int, w0: float, decay_updates: int) -> float`. Returns `w0 * max(0.0, 1.0 - n_updates / decay_updates)`.
  - Settings: `IL_PRELOAD_ROOT`; `IL_CRITIC_WARMUP_UPDATES` (default 5000); `IL_BC_WEIGHT0` (1.0); `IL_BC_DECAY_UPDATES` (10000).
  - Training flow:
    1. Before the first env step: fill the main replay with `which="exec"` and a separate expert buffer with `which="label"`.
    2. `set_warmup(True)`, run the warm-up updates, then `set_warmup(False)`.
    3. Online: after each `agent.update`, call `bc_step(agent, opt_bc, expert_chunk, bc_weight(...))`.
  - The cloning optimiser is held by the trainer, not stored in `agent.save`, so the checkpoint format is unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_warmup_freezes_encoder_core_and_actor():
    agent.set_warmup(True); before = snapshot(agent); agent.update(chunk)
    assert unchanged(before, agent, ("compress", "core", "actor")) and changed(before, agent, ("critics_g",))
def test_warmup_off_trains_everything_again(): ...
def test_bc_weight_schedule():
    assert bc_weight(0, 1.0, 10000) == 1.0 and bc_weight(5000, 1.0, 10000) == 0.5 and bc_weight(20000, 1.0, 10000) == 0.0
def test_checkpoint_format_unchanged(tmp_path):  # save after a bc_step -> loads strictly into a fresh agent
```
- [ ] **Step 2: Run them; expect failures**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests and the full suite; expect PASS and no new failures**
- [ ] **Step 5: Commit** `git commit -m "Fine-tune: frozen-encoder critic warm-up, decaying cloning loss"`

### Task 11: Chained goals 5-25 m ahead, chain eval, fine-tune runs

**Files:**
- Modify: `src/vae_nav/goal_chain.py`, `src/vae_nav/gated_reward.py`, `train_carla_sac.py` (`run_eval`), `torch_rl/carla_env_build.py`
- Create: `slurm/il_finetune.env`, `slurm/il_eval_chain.env`
- Test: `tests/test_goal_chain.py`, `tests/test_goal_zone_chain_reward.py`

**Interfaces:**
- Produces:
  - `goal_chain.goal_ahead(world_map, location, rng, min_m: float = 5.0, max_m: float = 25.0) -> tuple`.
    - Returns `(wp, stats)`.
    - Walks a uniformly drawn 5-25 m along the lane, picking a random branch at
      forks. Never ends inside a junction; `(None, stats)` at a dead end.
  - Setting `GOAL_CHAIN_STYLE`: `past_junction` (default; v35-v39 unchanged) or `ahead`.
  - Setting `EVAL_CHAIN=1`.
    - `build_env` passes `chain=True` in eval.
    - In `run_eval`, an arrival triggers `env.reset()` (no seed), so the chain continues.
    - A chain ends on a fault, the clock, or 200 m driven.
    - Each row records `chain_goals`, `chain_m`, `chain_end` and `chain_complete` (chain_m ≥ 200).

- [ ] **Step 1: Write the failing tests**

```python
def test_goal_ahead_lands_in_band_and_outside_junctions(): ...  # fake fork graph: 20 draws, distance in [5, 27]
def test_goal_ahead_draws_both_branches(): ...
def test_chain_style_ahead_is_used_on_soft_reset(monkeypatch): ...  # gated_reward calls goal_ahead
```
- [ ] **Step 2: Run them; expect failures**
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run the tests and the full suite; expect PASS and no new failures**

- [ ] **Step 5: Write `slurm/il_finetune.env`**

```bash
. "$REPO/slurm/il_expert.env"
export HER=1 REPLAY_CAPACITY=1000000 REPLAY_DEVICE=cpu REPLAY_U8=1 DET_EPISODE_FRAC=0.15 VIDEO_EVERY=20
export INIT_FROM=<selected clone> LEARNING_STARTS=20000
export IL_PRELOAD_ROOT=$HOME/carla-rl/checkpoints/il_data IL_CRITIC_WARMUP_UPDATES=5000 IL_BC_WEIGHT0=1.0 IL_BC_DECAY_UPDATES=10000
export GOAL_CHAIN_M=200 GOAL_CHAIN_STYLE=ahead GOAL_CHAIN_MIN_M=5 GOAL_CHAIN_MAX_M=25
export STALL_TRUNCATE_STEPS=100 STALL_SPEED_KMH=0.5
```

- [ ] **Step 6: Write `slurm/il_eval_chain.env`:** `. il_eval.env` plus `export EVAL_CHAIN=1 GOAL_CHAIN_M=200 GOAL_CHAIN_STYLE=ahead GOAL_CHAIN_MIN_M=5 GOAL_CHAIN_MAX_M=25`.

- [ ] **Step 7: Launch two seeds**
  - `SEED=0` and `SEED=1`, with `ENV_TOWNS=Town01,Town04,Town05,Town10HD PARALLEL_TOWNS=1`.
  - If four servers will not fit on one L40, fall back to two towns per run and say so.
  - Prove both alive.
  - Expected: `[Ep` lines led by `policy=`, and no encoder collapse (`enc_alive` near 1).

- [ ] **Step 8: Select and evaluate each seed**
  - Select each seed's checkpoint with Task 7's validation rule.
  - Then evaluate Town02 and Town03 (n=100 paired), the explore traces, and the chain eval (`il_eval_chain.env`, n=100).
  - Expected: one row per seed against every line of the success bar.

- [ ] **Step 9: Commit** `git commit -m "Chained goals 5-25 m ahead; chain eval; fine-tune overlay"`
