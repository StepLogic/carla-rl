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
from typing import Callable, Dict, Optional, Sequence

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
    expert: str = ""                    # FastRLAP's expert replay (torch_rl/frl/expert.py); "" = none
    actor: str = "frl"                  # "frl" = learner C's own actor; "c0" = c0's head (C0HeadActor)
    actor_freeze: int = 0               # training steps after start_training with critic-only updates
    random_start: int = 1               # 1 = uniform-random actions before start_training (FastRLAP)
    # How the expert replay is used: "update" = FastRLAP's second update per step on a separate expert
    # batch; "pinned" = the expert rows sit in the main replay, never overwritten, sampled uniformly with
    # the RL's own rows, the temperature tuned on the RL's own rows only (user 2026-11-11).
    expert_mode: str = "update"
    # 1 = every town's env stepped AT ONCE (train_parallel; user 2026-11-12 "build it"); 0 = one env at a
    # time, the next town at every episode end (train).
    parallel: int = 0

    def __post_init__(self):
        if self.expert_mode not in ("update", "pinned"):
            raise ValueError(f"expert_mode={self.expert_mode!r}: 'update' or 'pinned'")

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
    ONE env at a time -- one env per town, the next town at every episode end (train's
    extra_envs) -- trains from scratch with no demos, HER or noise-free episodes, and its trunk
    is c0's (MONO_CHW=1, SPEED_OBS=1 rows)."""
    env = os.environ if env is None else env
    out = []
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
    if str(env.get("FRL_EXPERT_MODE", "")).strip() == "pinned" and not env.get("FRL_EXPERT"):
        out.append("FRL_EXPERT_MODE=pinned with no FRL_EXPERT: there are no expert rows to pin")
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
    uniformly with replacement); embeddings stored float16, like the offline cache.

    pinned: expert transitions (torch_rl/frl/expert.ExpertReplay's arrays) held in the first rows,
    NEVER overwritten (the prerecorded-data rule); the RL's own transitions fill a ring of
    `capacity` rows after them. Sampling is uniform over all rows held; each batch marks the
    RL's own rows in "online" (bool), which is what the temperature is tuned on."""

    def __init__(self, capacity: int, emb_dim: int, state_dim: int, action_dim: int, device,
                 seed: int = 0, pinned: Optional[Dict] = None):
        dev = torch.device(device)
        c = int(capacity)
        p = 0 if pinned is None else int(pinned["rewards"].shape[0])
        self.capacity, self.n_pinned = c, p
        rows = p + c
        self.emb = torch.zeros(rows, emb_dim, dtype=torch.float16, device=dev)
        self.next_emb = torch.zeros(rows, emb_dim, dtype=torch.float16, device=dev)
        self.st = torch.zeros(rows, state_dim, device=dev)
        self.next_st = torch.zeros(rows, state_dim, device=dev)
        self.act = torch.zeros(rows, action_dim, device=dev)
        self.rew = torch.zeros(rows, device=dev)
        self.mask = torch.zeros(rows, device=dev)
        if p:
            for dst, key in ((self.emb, "emb"), (self.next_emb, "next_emb"), (self.st, "states"),
                             (self.next_st, "next_states"), (self.act, "actions"),
                             (self.rew, "rewards"), (self.mask, "masks")):
                src = torch.as_tensor(pinned[key])
                if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
                    raise ValueError(f"pinned {key} rows {tuple(src.shape[1:])} != the replay's "
                                     f"{tuple(dst.shape[1:])}")
                dst[:p] = src.to(dev, dst.dtype)
        self.n = self.i = 0                          # the RL's own rows held / next ring slot
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_pinned + self.n

    def add(self, obs, action, reward: float, mask: float, next_obs) -> None:
        i = self.n_pinned + self.i
        self.emb[i] = obs["embeddings"][0]
        self.st[i] = obs["states"][0]
        self.next_emb[i] = next_obs["embeddings"][0]
        self.next_st[i] = next_obs["states"][0]
        self.act[i] = torch.as_tensor(np.asarray(action, np.float32))
        self.rew[i] = float(reward)
        self.mask[i] = float(mask)
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int) -> Dict:
        idx = torch.as_tensor(self.rng.integers(0, len(self), size=int(batch)),
                              device=self.emb.device)
        return {"observations": {"embeddings": self.emb[idx].float(), "states": self.st[idx]},
                "next_observations": {"embeddings": self.next_emb[idx].float(),
                                      "states": self.next_st[idx]},
                "actions": self.act[idx], "rewards": self.rew[idx], "masks": self.mask[idx],
                "online": idx >= self.n_pinned}


class C0HeadActor(torch.nn.Module):
    """c0's own actor head -- a TRAINABLE COPY -- as learner C's actor (user 2026-11-11: "but the actor
    shares same frontend"). It reads c0's input, [GRU | mode, speed | binoc], uncompressed, in c0's order,
    so at step 0 the RL policy is c0 exactly; the critic keeps learner C's own (compressed) view."""

    def __init__(self, c0_actor: torch.nn.Module, hidden_dim: int):
        super().__init__()
        import copy
        self.net = copy.deepcopy(c0_actor)
        for p in self.net.parameters():
            p.requires_grad_(True)
        self.hidden_dim = int(hidden_dim)

    def c0_input(self, obs) -> torch.Tensor:
        e, st = obs["embeddings"].float(), obs["states"].float()
        return torch.cat([e[:, :self.hidden_dim], st, e[:, self.hidden_dim:]], dim=-1)

    def sample(self, obs):
        """-> (action, log pi(action), deterministic action tanh(mean)), learner C's actor API."""
        x = self.c0_input(obs)
        a, logp = self.net.sample(x)
        return a, logp, torch.tanh(self.net(x)[0])


def make_learner(emb_dim: int, state_dim: int, action_dim: int, device, seed: int = 0,
                 actor: str = "frl", trunk: Optional[C0Trunk] = None):
    """Learner C. actor="c0": its actor is replaced by a trainable copy of c0's head (C0HeadActor)."""
    from torch_rl.frl.learner import SACLearner
    learner = SACLearner(action_dim=action_dim, embedding_dim=emb_dim, state_dim=state_dim,
                         latent_dim=LATENT_DIM, seed=seed, device=str(device), **REDQ)
    if actor == "c0":
        if trunk is None:
            raise ValueError("actor='c0' needs the c0 trunk (its head is copied from it)")
        learner.actor = C0HeadActor(trunk.agent.actor, trunk.hidden_dim).to(learner.device)
        learner.actor_opt = torch.optim.Adam(learner.actor.parameters(), lr=REDQ["actor_lr"])
    elif actor != "frl":
        raise ValueError(f"actor={actor!r}: 'frl' or 'c0'")
    learner.dims = {"emb_dim": int(emb_dim), "state_dim": int(state_dim),
                    "action_dim": int(action_dim), "actor": actor}
    return learner


def critic_only_update(learner, batch: Dict, utd: int) -> Dict:
    """The critic part of learner.update alone -- utd minibatches, Polyak after each -- with the actor and
    the temperature untouched: the warm start's freeze, while a fresh critic learns c0's values."""
    from torch_rl.frl.learner import _slice, _to
    batch = _to(batch, learner.device)
    n = int(batch["rewards"].shape[0])
    bs = n // int(utd)
    info = {}
    for i in range(int(utd)):
        info = learner.update_critic(_slice(batch, i * bs, (i + 1) * bs))
    return info


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
    learner = make_learner(meta["emb_dim"], meta["state_dim"], meta["action_dim"], device,
                           actor=meta.get("actor", "frl"), trunk=trunk)
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
          to_env: Callable = box_map, ramp=None, rebuild_env: Optional[Callable] = None,
          expert=None, learner=None, extra_envs: Sequence = (),
          towns: Optional[Sequence[str]] = None):
    """FastRLAP's online loop on the CARLA env; returns (learner, replay, [the live env of each
    town]). `ramp` (a torch_rl.junction_ramp.JunctionRamp or None): the task's junction
    curriculum, fed every finished GOAL episode the actor drove, as the parallel loop feeds it.
    `rebuild_env` (None = a failed reset stops the run): builds a fresh env when a reset fails --
    after a simulator fault the actors can be gone, and only a rebuild recovers
    (vec_env._safe_reset).

    `extra_envs` [(env, rebuild or None), ...] (user 2026-11-11, "town05 and town01 train"):
    more towns, one env each. The loop still drives ONE env at a time -- one transition and one
    update per step, as with one town -- and moves to the next town at every episode end; a
    failed step restarts the episode in the same town. `towns` names the envs for the log."""
    dead = []        # KEPT REFERENCED: a collected CARLA client can std::terminate (vec_env)
    envs = [env] + [e for e, _ in extra_envs]
    rebuilds = [rebuild_env] + [r for _, r in extra_envs]
    names = list(towns) if towns is not None else [f"env{i}" for i in range(len(envs))]
    if len(names) != len(envs):
        raise ValueError(f"towns {names} name {len(names)} envs; there are {len(envs)}")
    cur = 0                                  # the town being driven

    def safe_reset():
        nonlocal env
        try:
            return env.reset()
        except Exception as e:  # noqa: BLE001
            if rebuilds[cur] is None:
                raise
            where = f" ({names[cur]})" if len(envs) > 1 else ""
            log(f"[Env] reset failed{where}: {type(e).__name__}: {e} -- rebuilding the env")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        dead.append(env)
        env = envs[cur] = rebuilds[cur]()
        return env.reset()

    s = settings
    box = env.action_space
    act_dim = int(box.shape[0])
    if learner is None:
        learner = make_learner(trunk.emb_dim, trunk.state_dim, act_dim, s.device, s.seed,
                               actor=s.actor, trunk=trunk)
    pinned = expert is not None and s.expert_mode == "pinned"
    # pinned: the expert rows sit in THIS replay (never overwritten) and there is no second update
    rb = FeatureReplay(s.replay, trunk.emb_dim, trunk.state_dim, act_dim,
                       s.replay_device or s.device, s.seed, pinned=expert.t if pinned else None)
    rng = np.random.default_rng(s.seed)
    torch.manual_seed(s.seed)
    hidden = trunk.zero_hidden()
    o_t, hidden = trunk.step(row_of(obs, info), hidden)
    mode = _mode(obs)
    ep_ret, ep_len, episodes = 0.0, 0, 0
    recent = {"policy": [], "goal": [], "explore": []}
    town_goal = {n: [] for n in names}       # the last 100 goal arrivals of each town
    town_eps = {n: 0 for n in names}
    ends, roll = {}, {"n": 0, "lock": 0, "idle_thr": 0}
    upd, t0 = {}, time.time()
    for step in range(1, s.max_steps + 1):
        learner.target_entropy -= s.entropy_anneal
        if step < s.start_training and s.random_start:
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
                town_eps[names[cur]] += 1
                if mode == 1:
                    town_goal[names[cur]] = (town_goal[names[cur]] + [arrived])[-100:]
                if ramp is not None and mode == 1 and step >= s.start_training:
                    msg = ramp.record(bool(arrived))
                    if msg:
                        log(msg)
                if len(envs) > 1:            # the next town drives the next episode
                    cur = (cur + 1) % len(envs)
                    env = envs[cur]
                obs, info = safe_reset()
                hidden = trunk.zero_hidden()
                o_n, hidden = trunk.step(row_of(obs, info), hidden)
                mode, ep_ret, ep_len = _mode(obs), 0.0, 0
            o_t = o_n
        if step >= s.start_training:
            frozen = step < s.start_training + s.actor_freeze
            if frozen:                    # warm start: the critic learns alone first
                upd = critic_only_update(learner, rb.sample(s.batch), s.utd)
            else:
                upd = learner.update(rb.sample(s.batch), utd_ratio=s.utd)
            if expert is not None and not pinned:
                # FastRLAP: a second update on the fixed expert replay, temperature untouched
                from torch_rl.frl import expert as _fx
                xinfo = (critic_only_update(learner, expert.sample(s.batch), s.utd) if frozen
                         else _fx.expert_update(learner, expert, s.batch, s.utd))
                upd.update({f"x_{key}": xinfo[key] for key in ("critic_loss", "q", "actor_loss")
                            if key in xinfo})
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
                f"{step / max(1e-9, time.time() - t0):.1f} steps/s"
                + (" | towns " + " ".join(
                    f"{n} goal={float(np.mean(town_goal[n])) if town_goal[n] else float('nan'):.3f}"
                    f"(n={len(town_goal[n])}) episodes={town_eps[n]}" for n in names)
                   if len(envs) > 1 else ""))
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
    return learner, rb, envs


# ---- the parallel loop (user 2026-11-12: "build it", "can we add more towns") ---------------------------
# Its learning, log line and checkpoints MIRROR train()'s above, which stays unchanged for the runs made
# with it; the helpers below are used by train_parallel only.

def _learn_step(learner, rb, expert, pinned: bool, s: LoopSettings, step: int) -> Dict:
    """One transition's learning, as train() does it: one update of s.batch split into s.utd
    minibatches; critic-only while the actor is frozen; FastRLAP's second expert update in "update"
    mode (none when the expert rows are pinned in the replay)."""
    frozen = step < s.start_training + s.actor_freeze
    if frozen:
        upd = critic_only_update(learner, rb.sample(s.batch), s.utd)
    else:
        upd = learner.update(rb.sample(s.batch), utd_ratio=s.utd)
    if expert is not None and not pinned:
        from torch_rl.frl import expert as _fx
        xinfo = (critic_only_update(learner, expert.sample(s.batch), s.utd) if frozen
                 else _fx.expert_update(learner, expert, s.batch, s.utd))
        upd.update({f"x_{key}": xinfo[key] for key in ("critic_loss", "q", "actor_loss")
                    if key in xinfo})
    return upd


def _checkpoint(out: str, step: int, learner, c0_dir: str, s: LoopSettings, probe, log) -> None:
    d = save_checkpoint(os.path.join(out, f"step_{step}"), learner, step=step, c0_dir=c0_dir,
                        settings=s)
    log(f"[Ckpt] {d}")
    if probe is not None:
        res = probe(learner.critic)
        with open(os.path.join(out, "probe.jsonl"), "a") as fh:
            fh.write(json.dumps({"step": step, **res}) + "\n")
        log(f"[Probe] step={step} " + " | ".join(
            f"{k[6:]} corr={v['all']['corr']} full_lock={v['all']['share_full_lock']} "
            f"|best|={v['all']['mean_abs_best']} (expert {v['all']['mean_abs_expert']})"
            for k, v in res.items()))


def train_parallel(env, obs, info, out: str, *, trunk: C0Trunk, row_of: Callable,
                   settings: LoopSettings, c0_dir: str, log: Callable = print,
                   probe: Optional[SteerProbe] = None, to_env: Callable = box_map, ramp=None,
                   rebuild_env: Optional[Callable] = None, expert=None, learner=None,
                   extra_envs: Sequence = (), towns: Optional[Sequence[str]] = None):
    """train() with every town's env STEPPED AT ONCE on a thread pool (one CARLA server each; the
    client mostly waits on its socket, so N envs cost about one round trip -- vae_nav.vec_env).
    Per transition it is train(): each env step is one transition, and each step at or past
    start_training is followed by one update (s.batch split into s.utd minibatches); the entropy
    target anneals per transition; max_steps, log_every and ckpt_every count transitions, and a
    checkpoint or log line is made when the count PASSES a multiple, under that multiple's name
    (step_25000), so evals find it. The actions of one iteration come from one batched forward of the
    same actor. Each env keeps its own GRU state and episode; a failed step abandons that env's
    episode only (nothing stored, that env reset, rebuilt if its reset fails). Returns (learner,
    replay, [the live env of each town])."""
    from concurrent.futures import ThreadPoolExecutor
    dead = []        # KEPT REFERENCED: a collected CARLA client can std::terminate (vec_env)
    envs = [env] + [e for e, _ in extra_envs]
    rebuilds = [rebuild_env] + [r for _, r in extra_envs]
    names = list(towns) if towns is not None else [f"env{i}" for i in range(len(envs))]
    if len(names) != len(envs):
        raise ValueError(f"towns {names} name {len(names)} envs; there are {len(envs)}")
    n = len(envs)
    s = settings
    box = env.action_space
    act_dim = int(box.shape[0])
    if learner is None:
        learner = make_learner(trunk.emb_dim, trunk.state_dim, act_dim, s.device, s.seed,
                               actor=s.actor, trunk=trunk)
    pinned = expert is not None and s.expert_mode == "pinned"
    rb = FeatureReplay(s.replay, trunk.emb_dim, trunk.state_dim, act_dim,
                       s.replay_device or s.device, s.seed, pinned=expert.t if pinned else None)
    rng = np.random.default_rng(s.seed)
    torch.manual_seed(s.seed)
    pool = ThreadPoolExecutor(max_workers=n)

    def safe_reset(i):
        try:
            return envs[i].reset()
        except Exception as e:  # noqa: BLE001
            if rebuilds[i] is None:
                raise
            log(f"[Env] reset failed ({names[i]}): {type(e).__name__}: {e} -- rebuilding the env")
        try:
            envs[i].close()
        except Exception:  # noqa: BLE001
            pass
        dead.append(envs[i])
        envs[i] = rebuilds[i]()
        return envs[i].reset()

    def env_step(i, a):
        try:
            return envs[i].step(a), None
        except Exception as e:  # noqa: BLE001 -- CARLA sensor timeouts and the like
            return None, e

    try:
        starts = [(obs, info)] + list(pool.map(safe_reset, range(1, n)))
        o_t, hidden, mode = [None] * n, [None] * n, [0] * n
        for i, (ob, inf) in enumerate(starts):
            o_t[i], hidden[i] = trunk.step(row_of(ob, inf), trunk.zero_hidden())
            mode[i] = _mode(ob)
        ep_ret, ep_len, episodes = [0.0] * n, [0] * n, 0
        recent = {"policy": [], "goal": [], "explore": []}
        town_goal = {nm: [] for nm in names}
        town_eps = {nm: 0 for nm in names}
        ends, roll = {}, {"n": 0, "lock": 0, "idle_thr": 0}
        upd, t0 = {}, time.time()
        step, next_log, next_ckpt, last_ckpt = 0, s.log_every, s.ckpt_every, 0
        while step < s.max_steps:
            # one batched forward of the actor for the envs past the random start
            acts = [None] * n
            pol = [i for i in range(n)
                   if not (step + 1 + i < s.start_training and s.random_start)]
            if pol:
                a_pol = learner.sample_actions(
                    {k: torch.cat([o_t[i][k] for i in pol], 0) for k in o_t[0]})
                for j, i in enumerate(pol):
                    acts[i] = np.asarray(a_pol[j], np.float32)
            for i in range(n):
                if acts[i] is None:
                    acts[i] = rng.uniform(-1.0, 1.0, act_dim).astype(np.float32)
            a_env = [to_env(a, box) for a in acts]
            for a, ae in zip(acts, a_env):
                roll["n"] += 1
                roll["lock"] += int(abs(float(a[0])) > 0.95)
                roll["idle_thr"] += int(act_dim > 1 and float(ae[1]) < 0.05)
            results = list(pool.map(env_step, range(n), a_env))
            need_reset, learn_at = [], []
            for i, (res, err) in enumerate(results):
                step += 1
                learner.target_entropy -= s.entropy_anneal
                if err is not None:
                    log(f"[Env] step {step} failed ({names[i]}): {type(err).__name__}: {err} -- "
                        f"episode abandoned after {ep_len[i]} real steps, env reset, nothing stored")
                    ends["env_error"] = ends.get("env_error", 0) + 1
                    need_reset.append(i)
                else:
                    ob, r, term, trunc, inf = res
                    o_n, hidden[i] = trunk.step(row_of(ob, inf), hidden[i])
                    rb.add(o_t[i], acts[i], float(r), 0.0 if term else 1.0, o_n)
                    ep_ret[i] += float(r)
                    ep_len[i] += 1
                    o_t[i] = o_n
                    if term or trunc:
                        episodes += 1
                        end = str(inf.get("goal_end") or inf.get("explore_end") or "?")
                        ends[end] = ends.get(end, 0) + 1
                        arrived = float(end == "reached")
                        recent["policy"].append(arrived)
                        recent["goal" if mode[i] == 1 else "explore"].append(
                            (arrived, ep_len[i], ep_ret[i]))
                        for k in recent:
                            recent[k] = recent[k][-100:]
                        town_eps[names[i]] += 1
                        if mode[i] == 1:
                            town_goal[names[i]] = (town_goal[names[i]] + [arrived])[-100:]
                        if ramp is not None and mode[i] == 1 and step >= s.start_training:
                            msg = ramp.record(bool(arrived))
                            if msg:
                                log(msg)
                        need_reset.append(i)
                if step >= s.start_training:
                    learn_at.append(step)
            if need_reset:
                for i, (ob, inf) in zip(need_reset, list(pool.map(safe_reset, need_reset))):
                    o_t[i], hidden[i] = trunk.step(row_of(ob, inf), trunk.zero_hidden())
                    mode[i], ep_ret[i], ep_len[i] = _mode(ob), 0.0, 0
            for k in learn_at:
                upd = _learn_step(learner, rb, expert, pinned, s, k)
            while step >= next_log:
                g, e = recent["goal"], recent["explore"]
                mean = lambda xs, j: float(np.mean([x[j] for x in xs])) if xs else float("nan")
                pol_sr = float(np.mean(recent["policy"])) if recent["policy"] else float("nan")
                log(f"[FRL] step={next_log} policy={pol_sr:.3f}(n={len(recent['policy'])}) "
                    f"goal={mean(g, 0):.3f}(n={len(g)}) goal_len={mean(g, 1):.0f} "
                    f"explore_len={mean(e, 1):.0f} episodes={episodes} ends={ends} | "
                    + " ".join(f"{k}={v:.4g}" for k, v in sorted(upd.items()))
                    + f" target_entropy={learner.target_entropy:.4f} | rollout steer_lock="
                    f"{roll['lock'] / max(1, roll['n']):.2f} throttle<0.05="
                    f"{roll['idle_thr'] / max(1, roll['n']):.2f} | "
                    f"{step / max(1e-9, time.time() - t0):.1f} steps/s"
                    + " | towns " + " ".join(
                        f"{nm} goal={float(np.mean(town_goal[nm])) if town_goal[nm] else float('nan'):.3f}"
                        f"(n={len(town_goal[nm])}) episodes={town_eps[nm]}"
                        for nm in dict.fromkeys(names)))
                ends, roll = {}, {"n": 0, "lock": 0, "idle_thr": 0}
                next_log += s.log_every
            while step >= next_ckpt and next_ckpt <= s.max_steps:
                _checkpoint(out, next_ckpt, learner, c0_dir, s, probe, log)
                last_ckpt, next_ckpt = next_ckpt, next_ckpt + s.ckpt_every
        if last_ckpt != s.max_steps:
            _checkpoint(out, s.max_steps, learner, c0_dir, s, probe, log)
    finally:
        pool.shutdown(wait=False)
    return learner, rb, envs


def run_carla(env, obs, info, server, out: str, *, row_of: Callable, to_env: Callable,
              rebuild_env: Optional[Callable] = None,
              log: Callable = lambda m: print(m, flush=True), extra_envs: Sequence = (),
              extra_servers: Sequence = (), towns: Optional[Sequence[str]] = None):
    """train_carla_sac.main's LEARNER=frl branch (after frl_refusals): preflight, train, close.
    The junction curriculum starts HERE, and the first episode is drawn again under it.
    `extra_envs`/`extra_servers`/`towns`: more towns, one env and one server each (train)."""
    s = LoopSettings.from_env()
    c0_dir = os.environ.get("FRL_C0", "")
    if not c0_dir:
        raise SystemExit("LEARNER=frl needs FRL_C0: c0's checkpoint directory (frozen trunk)")
    os.makedirs(out, exist_ok=True)
    trunk = C0Trunk(c0_dir, s.device)
    cache = os.environ.get("FRL_PROBE_CACHE", "")
    probe = SteerProbe(cache, s.device) if cache else None
    box = env.action_space
    expert = None
    if s.expert:
        from torch_rl.carla_env_build import action_repeat
        from torch_rl.frl.expert import ExpertReplay
        expert = ExpertReplay.load(s.expert, seed=s.seed)
        if expert.k != action_repeat():
            raise SystemExit(f"FRL_EXPERT {s.expert} was built at action repeat {expert.k}; this run "
                             f"acts at ACTION_REPEAT={action_repeat()}")
        if expert.c0_md5 != params_md5(c0_dir):
            raise SystemExit(f"FRL_EXPERT {s.expert} was built on c0 md5 {expert.c0_md5}; this run's "
                             f"trunk is {params_md5(c0_dir)}")
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
    log(f"  ACTOR: {'c0 head (trainable copy; the RL policy starts as c0)' if s.actor == 'c0' else 'learner C (from scratch)'}"
        f" | critic-only steps after start_training: {s.actor_freeze} | random start: "
        f"{'on' if s.random_start else 'OFF (the actor drives from step 1)'}")
    how = ({"update": "FastRLAP's second update per step on an expert batch, temperature untouched",
            "pinned": "PINNED in the main replay: never overwritten, sampled uniformly with the RL's own "
                      "rows, no second update, temperature tuned on the RL's own rows only"}
           [s.expert_mode])
    log(f"  EXPERT REPLAY ({how}): "
        + (f"{s.expert}: {len(expert)} transitions at k={expert.k}, {expert.meta}" if expert
           else "OFF"))
    log(f"  probe: {cache or 'OFF'} | checkpoints -> {out}")
    if s.parallel:
        log(f"  TOWNS: {', '.join(towns or ['(one)'])} -- one env and one CARLA server each, all "
            f"stepped IN PARALLEL (thread pool); per transition as train(): one update per "
            f"transition, the entropy target annealed per transition; replay on {s.replay_device or s.device}")
    elif extra_envs:
        log(f"  TOWNS: {', '.join(towns or [])} -- one env and one CARLA server each; the loop "
            f"drives one at a time and moves to the next town at every episode end (one "
            f"transition and one update per step, as with one town)")
    t0 = time.time()
    live = [env] + [e for e, _ in extra_envs]
    loop = train_parallel if s.parallel else train
    try:
        _, _, live = loop(env, obs, info, out, trunk=trunk, row_of=row_of, settings=s,
                          c0_dir=c0_dir, log=log, probe=probe, to_env=to_env, ramp=ramp,
                          rebuild_env=rebuild_env, expert=expert, extra_envs=extra_envs,
                          towns=towns)
    finally:
        for e in live:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        for srv in (server, *extra_servers):
            if srv is not None:
                srv.stop()
    log(f"[Done] {s.max_steps} steps in {(time.time() - t0) / 3600:.2f} h -> {out}")
