"""Learner C in CARLA (user 2026-11-10: "Compressed to 50-d, compression trained by the critic
(FastRLAP)", "proceed build it and run locally", "move straight to learner C").

FRONT END. c0's FROZEN trunk -- the imitation clone (il_bc_c0 step_15000): DEBiT-initialised
monocular ResNet and GRU -- runs online, one GRU step per env step from a zero state at every
episode start, exactly as GatedRecurrentSAC.act does (and as scripts/probe_c0_features.py built
the offline cache the steering probe reads). Its output h_t (512) and the env's frozen DEBiT
binocular vector (3136) are learner C's `embeddings`; [mode, speed] are its `states`.

LEARNER. torch_rl.frl.learner.SACLearner, FastRLAP's embeddings mode: each input group through
its own Dense(50) -> LayerNorm -> tanh (the critic's trained by the critic loss, the actor's by
the actor loss), 10 LayerNorm Q heads, min over 2 target heads, no entropy term in the target,
the actor on the mean of all heads; actor, critics and temperature from scratch. Settings = the
benchmark's redq config (scripts/bench/frl_train.py CONFIGS["redq"], FastRLAP's redq_config).

LOOP. FastRLAP's (fastrlap-mujoco-env/train_online_pixels.py, as scripts/bench/frl_train.py):
uniform-random actions for the first `start_training` steps, then the actor's sampled action;
mask 0 only on a true termination (a time limit keeps bootstrapping); one update of 256
transitions split into utd 8 minibatches per env step; the target entropy annealed by -25e-7
per step. Nothing else: no HER, no noise-free episodes, no junction curriculum, no demos, no
warm start. The env (reward, terminals, goal sampler) is the run's overlay, unchanged.

CHECKPOINTS are step_<N>/params.pt + config.json ({"learner": "frl", ...}, written last). The
eval harness (train_carla_sac.run_eval) drives one through FrlC0Policy, which has
GatedRecurrentSAC's acting API (reset_hidden, act). At every checkpoint the steering probe of
the offline test (scripts/offline_priv_critic.py) reads the critic on the held-out Town05 demo
states (FRL_PROBE_CACHE): best steering on a 21-point grid vs the expert's.
"""
import dataclasses
import json
import os
import sys
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

# FastRLAP's redq_config as the benchmark ran it (pinned by tests/test_frl_carla.py).
REDQ = {"actor_lr": 3e-4, "critic_lr": 3e-4, "temp_lr": 3e-4, "hidden_dims": (256, 256),
        "discount": 0.99, "num_qs": 10, "num_min_qs": 2, "critic_layer_norm": True,
        "tau": 0.005, "init_temperature": 0.1, "backup_entropy": False}
LATENT_DIM = 50
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclasses.dataclass
class LoopSettings:
    max_steps: int = 100000
    start_training: int = 1000          # FastRLAP start_training
    utd: int = 8                        # FastRLAP utd_ratio
    batch: int = 256                    # FastRLAP batch_size (split into utd minibatches)
    replay: int = 100000
    ckpt_every: int = 25000
    log_every: int = 1000
    device: str = "cuda"
    replay_device: str = ""             # "" = device
    seed: int = 0
    entropy_anneal: float = 25e-7       # FastRLAP: target_entropy -= 25e-7 per step

    @classmethod
    def from_env(cls, env=None) -> "LoopSettings":
        """FRL_<FIELD> overrides; the device and seed otherwise follow the trainer's own
        TORCH_DEVICE and SEED, as the encoder, the env and the evals do."""
        env = os.environ if env is None else env
        fallback = {"device": "TORCH_DEVICE", "seed": "SEED"}
        kw = {}
        for f in dataclasses.fields(cls):
            raw = env.get("FRL_" + f.name.upper())
            if (raw is None or raw == "") and f.name in fallback:
                raw = env.get(fallback[f.name])
            if raw is not None and raw != "":
                kw[f.name] = type(f.default)(raw)
        return cls(**kw)


def frl_refusals(towns, parallel: bool, env=None) -> list:
    """Settings learner C's loop would silently ignore or get wrong; [] = none. Its loop drives
    ONE env with no town rotation, trains from scratch with no demos, HER or noise-free
    episodes, and its trunk is c0's (MONO_CHW=1, SPEED_OBS=1 rows)."""
    env = os.environ if env is None else env
    out = []
    if len(towns) != 1:
        out.append(f"ENV_TOWNS={','.join(towns)}: learner C trains on one town (no rotation)")
    if parallel:
        out.append("PARALLEL_TOWNS=1: learner C drives one env; set PARALLEL_TOWNS=0")
    for k in ("INIT_FROM", "IL_PRELOAD_ROOT"):
        if env.get(k):
            out.append(f"{k}={env[k]}: learner C trains from scratch without demos")
    if str(env.get("HER", "0")).strip() not in ("", "0"):
        out.append(f"HER={env['HER']}: learner C's loop has no hindsight relabelling; set HER=0")
    if float(env.get("DET_EPISODE_FRAC", "0") or 0.0) != 0.0:
        out.append(f"DET_EPISODE_FRAC={env['DET_EPISODE_FRAC']}: learner C's loop has no "
                   f"noise-free episodes; set 0")
    for k in ("MONO_CHW", "SPEED_OBS"):
        if str(env.get(k, "")).strip() != "1":
            out.append(f"{k}={env.get(k)}: c0's trunk reads {k}=1 rows")
    return out


def make_ramp(env=None):
    """The task's junction curriculum (torch_rl.junction_ramp) from JUNCTION_RAMP_*, as
    train_carla_sac.main builds it for the parallel loop; None when JUNCTION_RAMP_START is
    unset. Constructing it sets JUNCTION_GOAL_FRAC to the start value."""
    env = os.environ if env is None else env
    if not env.get("JUNCTION_RAMP_START"):
        return None
    from torch_rl.junction_ramp import JunctionRamp
    f = lambda k, d: float(env.get(k, d))  # noqa: E731
    return JunctionRamp(start=f("JUNCTION_RAMP_START", 0.5), target=f("JUNCTION_GOAL_FRAC", 0.85),
                        step=f("JUNCTION_RAMP_STEP", 0.05), up=f("JUNCTION_RAMP_UP", 0.3),
                        down=f("JUNCTION_RAMP_DOWN", 0.1),
                        window=int(f("JUNCTION_RAMP_WINDOW", 100)))


_MD5 = {}


def params_md5(ckpt_dir: str) -> str:
    """md5 of a checkpoint's params.pt (cached per path, size and mtime)."""
    import hashlib
    p = os.path.abspath(os.path.join(ckpt_dir, "params.pt"))
    st = os.stat(p)
    key = (p, st.st_size, st.st_mtime_ns)
    if key not in _MD5:
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        _MD5[key] = h.hexdigest()
    return _MD5[key]


def box_map(a, box):
    """Actor action in [-1, 1] -> the env's box (train_carla_sac.to_env)."""
    lo, hi = box.low.astype(np.float32), box.high.astype(np.float32)
    return lo + 0.5 * (np.clip(a, -1.0, 1.0) + 1.0) * (hi - lo)


class C0Trunk:
    """c0's frozen encoder + GRU as a feature extractor: row -> learner C's observation."""

    def __init__(self, c0_dir: str, device):
        from torch_rl import pixel_rows as pxr
        from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig
        with open(os.path.join(c0_dir, "config.json")) as fh:
            d = json.load(fh)
        d.update(device=str(device), compile=False, bf16=False)
        self.agent = GatedRecurrentSAC(GatedRecurrentSACConfig(**d))
        self.agent.load(c0_dir)                                   # strict: every module
        cfg = self.agent.cfg
        if cfg.mode_index != 0 or cfg.prev_action or cfg.goal_tokens:
            raise ValueError("C0Trunk expects c0's layout: mode first, no prev action, "
                             "no goal tokens")
        for m in (self.agent.compress, self.agent.core):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.c0_dir = c0_dir
        self.device = torch.device(device)
        self.hidden_dim = int(cfg.hidden)
        self.binoc_dim = int(pxr.BINOC_DIM)
        self.state_dim = int(cfg.state_dim) - self.binoc_dim     # [mode, speed]
        self.emb_dim = self.hidden_dim + self.binoc_dim

    def zero_hidden(self) -> torch.Tensor:
        return self.agent.reset_hidden(1)

    @torch.no_grad()
    def step(self, row, hidden):
        a = self.agent
        x = torch.as_tensor(np.asarray(row, np.float32)[None], device=self.device)
        s, hidden = a.core.step(a._embed(a.compress, a.act_embed, x), hidden)
        st = a._state(x).float()
        obs = {"embeddings": torch.cat([s.float(), st[:, self.state_dim:]], dim=-1),
               "states": st[:, :self.state_dim]}
        return obs, hidden


class FeatureReplay:
    """Uniform transition replay of learner C's inputs (FastRLAP's ReplayBuffer: sampled
    uniformly with replacement); embeddings stored float16, like the offline cache."""

    def __init__(self, capacity: int, emb_dim: int, state_dim: int, action_dim: int, device,
                 seed: int = 0):
        dev = torch.device(device)
        c = int(capacity)
        self.capacity = c
        self.emb = torch.zeros(c, emb_dim, dtype=torch.float16, device=dev)
        self.next_emb = torch.zeros(c, emb_dim, dtype=torch.float16, device=dev)
        self.st = torch.zeros(c, state_dim, device=dev)
        self.next_st = torch.zeros(c, state_dim, device=dev)
        self.act = torch.zeros(c, action_dim, device=dev)
        self.rew = torch.zeros(c, device=dev)
        self.mask = torch.zeros(c, device=dev)
        self.n = self.i = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def add(self, obs, action, reward: float, mask: float, next_obs) -> None:
        i = self.i
        self.emb[i] = obs["embeddings"][0]
        self.st[i] = obs["states"][0]
        self.next_emb[i] = next_obs["embeddings"][0]
        self.next_st[i] = next_obs["states"][0]
        self.act[i] = torch.as_tensor(np.asarray(action, np.float32))
        self.rew[i] = float(reward)
        self.mask[i] = float(mask)
        self.i = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int) -> Dict:
        idx = torch.as_tensor(self.rng.integers(0, self.n, size=int(batch)), device=self.emb.device)
        return {"observations": {"embeddings": self.emb[idx].float(), "states": self.st[idx]},
                "next_observations": {"embeddings": self.next_emb[idx].float(),
                                      "states": self.next_st[idx]},
                "actions": self.act[idx], "rewards": self.rew[idx], "masks": self.mask[idx]}


def make_learner(emb_dim: int, state_dim: int, action_dim: int, device, seed: int = 0):
    from torch_rl.frl.learner import SACLearner
    learner = SACLearner(action_dim=action_dim, embedding_dim=emb_dim, state_dim=state_dim,
                         latent_dim=LATENT_DIM, seed=seed, device=str(device), **REDQ)
    learner.dims = {"emb_dim": int(emb_dim), "state_dim": int(state_dim),
                    "action_dim": int(action_dim)}
    return learner


class FrlC0Policy:
    """Learner C's actor behind c0's frozen trunk, with GatedRecurrentSAC's acting API, so
    train_carla_sac.run_eval drives it unchanged. The hidden is c0's GRU state."""

    def __init__(self, trunk: C0Trunk, actor):
        self.trunk, self.actor = trunk, actor

    def reset_hidden(self, batch: int = 1) -> torch.Tensor:
        if int(batch) != 1:
            raise ValueError("FrlC0Policy acts on one env")
        return self.trunk.zero_hidden()

    @torch.no_grad()
    def act(self, row, hidden, deterministic: bool = False, goal=None, noise_scale: float = 1.0):
        if not deterministic and float(noise_scale) != 1.0:
            raise ValueError("learner C samples at its own noise; noise_scale must be 1 or the "
                             "action deterministic")
        obs, hidden = self.trunk.step(row, hidden)
        a, _, mean_a = self.actor.sample(obs)
        return (mean_a if deterministic else a)[0].cpu().numpy(), hidden


def save_checkpoint(d: str, learner, *, step: int, c0_dir: str, settings: LoopSettings) -> str:
    os.makedirs(d, exist_ok=True)
    torch.save({"actor": learner.actor.state_dict(), "critic": learner.critic.state_dict(),
                "target_critic": learner.target_critic.state_dict(),
                "log_temp": learner.log_temp.detach().cpu(),
                "target_entropy": float(learner.target_entropy),
                "actor_opt": learner.actor_opt.state_dict(),
                "critic_opt": learner.critic_opt.state_dict(),
                "temp_opt": learner.temp_opt.state_dict()}, os.path.join(d, "params.pt"))
    meta = {"learner": "frl", "step": int(step), "c0": os.path.abspath(c0_dir),
            "c0_md5": params_md5(c0_dir), "latent_dim": LATENT_DIM, **learner.dims,
            "redq": {k: list(v) if isinstance(v, tuple) else v for k, v in REDQ.items()},
            "settings": dataclasses.asdict(settings)}
    tmp = os.path.join(d, "config.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, os.path.join(d, "config.json"))   # last: marks a whole checkpoint
    return d


def is_frl_checkpoint(d: str) -> bool:
    try:
        with open(os.path.join(d, "config.json")) as fh:
            return json.load(fh).get("learner") == "frl"
    except (OSError, ValueError):
        return False


def load_policy(d: str, device) -> FrlC0Policy:
    with open(os.path.join(d, "config.json")) as fh:
        meta = json.load(fh)
    # THE TRUNK IS PART OF THE POLICY: the actor was trained on THIS c0's features. FRL_C0 may
    # only point at the same trunk elsewhere (e.g. another machine's copy), checked by md5.
    c0 = os.environ.get("FRL_C0") or meta["c0"]
    if params_md5(c0) != meta["c0_md5"]:
        raise ValueError(f"c0 trunk {c0} (md5 {params_md5(c0)}) is not the one this checkpoint "
                         f"was trained on ({meta['c0']}, md5 {meta['c0_md5']})")
    trunk = C0Trunk(c0, device)
    if (trunk.emb_dim, trunk.state_dim) != (meta["emb_dim"], meta["state_dim"]):
        raise ValueError(f"c0 trunk dims {(trunk.emb_dim, trunk.state_dim)} != the "
                         f"checkpoint's {(meta['emb_dim'], meta['state_dim'])}")
    learner = make_learner(meta["emb_dim"], meta["state_dim"], meta["action_dim"], device)
    state = torch.load(os.path.join(d, "params.pt"), map_location=str(device))
    learner.actor.load_state_dict(state["actor"])
    return FrlC0Policy(trunk, learner.actor)


class SteerProbe:
    """The offline steering test's probe on the online critic: the held-out Town05 demo states
    of scripts/offline_priv_critic.py (first 3 shards + every 10th), the expert's label action
    with its steering swept over 21 values; corr / full-lock share / mean |best| per set."""

    def __init__(self, cache: str, device):
        # APPENDED, never first on the path: a scripts/ module can then never shadow another
        # import. The run's MONO_CHW=1 / SPEED_OBS=1 (frl_refusals) make probe_c0_features'
        # import-time os.environ.setdefault of the same values a no-op.
        scripts = os.path.join(REPO, "scripts")
        if scripts not in sys.path:
            sys.path.append(scripts)
        import offline_priv_critic as opc
        from probe_c0_features import clean_label_mask
        self.opc, self.device = opc, str(device)
        C = dict(np.load(cache, allow_pickle=False))
        X, clean = opc.image_features(C), clean_label_mask(C)
        # only the held-out probe rows are kept for the run
        self.sets = {name: (np.asarray(X[fr]), C["s_act_label"][sp], C["f_junction"][fr],
                            clean[fr])
                     for name, (fr, sp) in opc.held_out_split(C)[1].items()}

    def __call__(self, critic) -> Dict:
        opc = self.opc
        return {name: opc.measure(opc.MeasureView(critic), x, lab, jn, cl, self.device)
                for name, (x, lab, jn, cl) in self.sets.items()}


def _mode(obs) -> int:
    return int(round(float(np.asarray(obs["mode"]).ravel()[0])))


def train(env, obs, info, out: str, *, trunk: C0Trunk, row_of: Callable, settings: LoopSettings,
          c0_dir: str, log: Callable = print, probe: Optional[SteerProbe] = None,
          to_env: Callable = box_map, ramp=None, rebuild_env: Optional[Callable] = None):
    """FastRLAP's online loop on the CARLA env; returns (learner, replay, the live env). `ramp`
    (a torch_rl.junction_ramp.JunctionRamp or None): the task's junction curriculum, fed every
    finished GOAL episode the actor drove, as the parallel loop feeds it. `rebuild_env` (None =
    a failed reset stops the run): builds a fresh env when a reset fails -- after a simulator
    fault the actors can be gone, and only a rebuild recovers (vec_env._safe_reset)."""
    dead = []        # KEPT REFERENCED: a collected CARLA client can std::terminate (vec_env)

    def safe_reset():
        nonlocal env
        try:
            return env.reset()
        except Exception as e:  # noqa: BLE001
            if rebuild_env is None:
                raise
            log(f"[Env] reset failed: {type(e).__name__}: {e} -- rebuilding the env")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        dead.append(env)
        env = rebuild_env()
        return env.reset()

    s = settings
    box = env.action_space
    act_dim = int(box.shape[0])
    learner = make_learner(trunk.emb_dim, trunk.state_dim, act_dim, s.device, s.seed)
    rb = FeatureReplay(s.replay, trunk.emb_dim, trunk.state_dim, act_dim,
                       s.replay_device or s.device, s.seed)
    rng = np.random.default_rng(s.seed)
    torch.manual_seed(s.seed)
    hidden = trunk.zero_hidden()
    o_t, hidden = trunk.step(row_of(obs, info), hidden)
    mode = _mode(obs)
    ep_ret, ep_len, episodes = 0.0, 0, 0
    recent = {"policy": [], "goal": [], "explore": []}
    ends, roll = {}, {"n": 0, "lock": 0, "idle_thr": 0}
    upd, t0 = {}, time.time()
    for step in range(1, s.max_steps + 1):
        learner.target_entropy -= s.entropy_anneal
        if step < s.start_training:
            a = rng.uniform(-1.0, 1.0, act_dim).astype(np.float32)
        else:
            a = learner.sample_actions(o_t)[0].astype(np.float32)
        a_env = to_env(a, box)
        roll["n"] += 1
        roll["lock"] += int(abs(float(a[0])) > 0.95)
        roll["idle_thr"] += int(act_dim > 1 and float(a_env[1]) < 0.05)
        try:
            obs, r, term, trunc, info = env.step(a_env)
        except Exception as e:  # noqa: BLE001 -- CARLA sensor timeouts and the like
            # A FAILED SIMULATOR STEP: there is no real transition, so none is stored; the
            # episode is abandoned and the next one starts (the parallel loop's env_error rule).
            log(f"[Env] step {step} failed: {type(e).__name__}: {e} -- episode abandoned after "
                f"{ep_len} real steps, env reset, nothing stored")
            ends["env_error"] = ends.get("env_error", 0) + 1
            obs, info = safe_reset()
            hidden = trunk.zero_hidden()
            o_t, hidden = trunk.step(row_of(obs, info), hidden)
            mode, ep_ret, ep_len = _mode(obs), 0.0, 0
        else:
            o_n, hidden = trunk.step(row_of(obs, info), hidden)
            rb.add(o_t, a, float(r), 0.0 if term else 1.0, o_n)
            ep_ret += float(r)
            ep_len += 1
            if term or trunc:
                episodes += 1
                end = str(info.get("goal_end") or info.get("explore_end") or "?")
                ends[end] = ends.get(end, 0) + 1
                arrived = float(end == "reached")
                recent["policy"].append(arrived)
                recent["goal" if mode == 1 else "explore"].append((arrived, ep_len, ep_ret))
                for k in recent:
                    recent[k] = recent[k][-100:]
                if ramp is not None and mode == 1 and step >= s.start_training:
                    msg = ramp.record(bool(arrived))
                    if msg:
                        log(msg)
                obs, info = safe_reset()
                hidden = trunk.zero_hidden()
                o_n, hidden = trunk.step(row_of(obs, info), hidden)
                mode, ep_ret, ep_len = _mode(obs), 0.0, 0
            o_t = o_n
        if step >= s.start_training:
            upd = learner.update(rb.sample(s.batch), utd_ratio=s.utd)
        if step % s.log_every == 0:
            g, e = recent["goal"], recent["explore"]
            mean = lambda xs, j: float(np.mean([x[j] for x in xs])) if xs else float("nan")
            pol = float(np.mean(recent["policy"])) if recent["policy"] else float("nan")
            log(f"[FRL] step={step} policy={pol:.3f}(n={len(recent['policy'])}) "
                f"goal={mean(g, 0):.3f}(n={len(g)}) goal_len={mean(g, 1):.0f} "
                f"explore_len={mean(e, 1):.0f} episodes={episodes} ends={ends} | "
                + " ".join(f"{k}={v:.4g}" for k, v in sorted(upd.items()))
                + f" target_entropy={learner.target_entropy:.4f} | rollout steer_lock="
                f"{roll['lock'] / max(1, roll['n']):.2f} throttle<0.05="
                f"{roll['idle_thr'] / max(1, roll['n']):.2f} | "
                f"{step / max(1e-9, time.time() - t0):.1f} steps/s")
            ends, roll = {}, {"n": 0, "lock": 0, "idle_thr": 0}
        if step % s.ckpt_every == 0 or step == s.max_steps:
            d = save_checkpoint(os.path.join(out, f"step_{step}"), learner, step=step,
                                c0_dir=c0_dir, settings=s)
            log(f"[Ckpt] {d}")
            if probe is not None:
                res = probe(learner.critic)
                row = {"step": step, **res}
                with open(os.path.join(out, "probe.jsonl"), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log(f"[Probe] step={step} " + " | ".join(
                    f"{k[6:]} corr={v['all']['corr']} full_lock={v['all']['share_full_lock']} "
                    f"|best|={v['all']['mean_abs_best']} (expert {v['all']['mean_abs_expert']})"
                    for k, v in res.items()))
    return learner, rb, env


def run_carla(env, obs, info, server, out: str, *, row_of: Callable, to_env: Callable,
              rebuild_env: Optional[Callable] = None,
              log: Callable = lambda m: print(m, flush=True)):
    """train_carla_sac.main's LEARNER=frl branch (after frl_refusals): preflight, train, close.
    The junction curriculum starts HERE, and the first episode is drawn again under it."""
    s = LoopSettings.from_env()
    c0_dir = os.environ.get("FRL_C0", "")
    if not c0_dir:
        raise SystemExit("LEARNER=frl needs FRL_C0: c0's checkpoint directory (frozen trunk)")
    os.makedirs(out, exist_ok=True)
    trunk = C0Trunk(c0_dir, s.device)
    cache = os.environ.get("FRL_PROBE_CACHE", "")
    probe = SteerProbe(cache, s.device) if cache else None
    box = env.action_space
    ramp = make_ramp()
    if ramp is not None:
        obs, info = env.reset()       # main drew the first goal at the target fraction
    e = os.environ.get
    log(f"=== [{e('RUN_NAME')}] preflight: LEARNER C (FastRLAP SACLearner, embeddings mode) ===")
    log(f"  front end: c0 FROZEN trunk {c0_dir} -> embeddings [GRU {trunk.hidden_dim} | binoc "
        f"{trunk.binoc_dim}] = {trunk.emb_dim}-d, states [mode, speed] = {trunk.state_dim}-d; "
        f"each -> Dense({LATENT_DIM}) -> LayerNorm -> tanh (critic's trained by the critic)")
    log(f"  learner: {REDQ} latent {LATENT_DIM}; actor/critics/temperature FROM SCRATCH")
    log(f"  loop: {dataclasses.asdict(s)}; random actions for steps < {s.start_training}; one "
        f"update of {s.batch} split into {s.utd} minibatches per step; mask 0 only on "
        f"termination. NOT used: HER, noise-free episodes, junction curriculum, demos, warm start")
    log(f"  env: towns={e('ENV_TOWNS')} port={e('CARLA_PORT')} quality={e('CARLA_QUALITY')} "
        f"offscreen={e('CARLA_OFFSCREEN')} | action box steer [{box.low[0]:+.2f}, "
        f"{box.high[0]:+.2f}] throttle [{box.low[1]:+.2f}, {box.high[1]:+.2f}]")
    log(f"  reward: REWARD_SCALE={e('REWARD_SCALE')} arrival bonus={e('GOAL_EXPLORE_ARRIVAL_BONUS')} "
        f"band={e('GOAL_DIST_MIN')}-{e('GOAL_DIST_MAX')} m radius={e('GOAL_RADIUS_M')} "
        f"arrival max speed={e('GOAL_ARRIVAL_MAX_SPEED')} junction frac="
        f"{e('JUNCTION_GOAL_FRAC')} stall truncate={e('STALL_TRUNCATE_STEPS')} lane mode="
        f"{e('EXPLORE_LANE_MODE')} off-road fatal={e('EXPLORE_OFF_ROAD_FATAL')}")
    log(f"  {ramp.describe() if ramp is not None else 'JUNCTION RAMP: off'} (episodes the "
        f"random warm-up drove are not counted)")
    log(f"  c0 trunk md5 {params_md5(c0_dir)} | the gated loop's settings are NOT read: "
        + " ".join(f"{k}={e(k)}" for k in ("MAX_STEPS", "CKPT_EVERY", "LEARNING_STARTS",
                                           "UPDATE_EVERY", "REPLAY_CAPACITY") if e(k)))
    log(f"  env rebuild on a failed reset: {'on' if rebuild_env else 'OFF'}")
    log(f"  probe: {cache or 'OFF'} | checkpoints -> {out}")
    t0 = time.time()
    live = env
    try:
        _, _, live = train(env, obs, info, out, trunk=trunk, row_of=row_of, settings=s,
                           c0_dir=c0_dir, log=log, probe=probe, to_env=to_env, ramp=ramp,
                           rebuild_env=rebuild_env)
    finally:
        try:
            live.close()
        except Exception:  # noqa: BLE001
            pass
        if server is not None:
            server.stop()
    log(f"[Done] {s.max_steps} steps in {(time.time() - t0) / 3600:.2f} h -> {out}")
