"""Learner C in CARLA (user 2026-11-10: "Compressed to 50-d, compression trained by the critic
(FastRLAP)", "proceed build it and run locally", "move straight to learner C").

FRONT END. c0's FROZEN trunk -- the imitation clone (il_bc_c0 step_15000): DEBiT-initialised
monocular ResNet and GRU -- runs online, one GRU step per env step from a zero state at every
episode start, exactly as GatedRecurrentSAC.act does (and as scripts/probe_c0_features.py built
the offline cache the steering probe reads). Its output h_t (512) and the env's frozen DEBiT
binocular vector (3136) are learner C's `embeddings`; [mode, speed] are its `states`.
With FRL_CONTROL (user 2026-11-14) the frozen trunk is the pretrained CONTROL STREAM instead
(ControlTrunk: a frame pair one decision apart -> small CNN -> GRU that also takes the speed),
whose state carries the heading error c0's features lack; the binocular vector is unchanged.

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
    # A checkpoint dir to resume from (train_parallel; user 2026-11-12 "resume current run with new code"):
    # the whole learner and the step count are restored; the replay is not in a checkpoint and starts fresh.
    resume: str = ""
    # 1 = the critic's heads computed in one batched pass (networks.EnsembleQ.batched): the same weights
    # and numbers, faster; a compute layout, so a resumed run may change it.
    batched_heads: int = 0
    # 1 = RESUME THE CRITIC BUT KEEP THIS RUN'S ACTOR (user 2026-11-13: "use saved checkpoint. and
    # replace actor with c0 actor"). restore_learner then loads the critic, the target critic, the
    # temperature, the entropy target and their optimizers, and leaves the actor `actor` built -- with
    # actor="c0", a trainable copy of c0's head, so the policy starts as c0 on a critic that already has
    # the run's value estimates. The actor's optimizer stays fresh: its parameters are a different shape,
    # so the checkpoint's Adam state does not apply. It also lets `actor` differ from the checkpoint's,
    # which the guard otherwise refuses.
    resume_new_actor: int = 0
    # THE CONTROL STREAM AS THE TRUNK (user 2026-11-14: "Queue online train for best fixes wirh SAC
    # with expert pinned rows"): a directory holding the pretrained torch_rl.control_cnn.ControlGRU
    # (params.pt from scripts/pretrain_control.py). Its frozen GRU state replaces c0's in the
    # embeddings (ControlTrunk); DEBiT's binocular features stay as they are. "" = c0's trunk
    # (FRL_C0), as every earlier run. The path may differ between machines: the stream's identity is
    # the md5 of its params.pt, kept in every checkpoint.
    control: str = ""
    # IQL updates on the expert rows BEFORE the first online step (user 2026-11-14: "also pretrain
    # critic with IQL first"; FastRLAP's iql_learner rules, torch_rl.frl.iql.iql_warmup) on this
    # learner's own critic and actor. 0 = none. Needs an expert replay; a resumed run skips it.
    iql_pretrain: int = 0

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


def warm_start_refusals(control: str, actor: str, iql_pretrain: int, expert: str) -> list:
    """What the control-stream trunk and the IQL warm-up cannot be combined with; [] = fine. Read
    from the environment before a server starts (frl_refusals) and again from the run's own
    settings (run_carla)."""
    out = []
    if control and actor == "c0":
        out.append("FRL_ACTOR=c0 with FRL_CONTROL: c0's head reads c0's trunk, not the control "
                   "stream's features")
    if int(iql_pretrain or 0) > 0 and not expert:
        out.append(f"FRL_IQL_PRETRAIN={iql_pretrain} with no FRL_EXPERT: the warm-up learns from "
                   f"the expert rows")
    if int(iql_pretrain or 0) > 0 and actor == "c0":
        out.append(f"FRL_IQL_PRETRAIN={iql_pretrain} with FRL_ACTOR=c0: the warm-up's AWR step needs "
                   f"the log-probability of a dataset action, which c0's head does not give")
    return out


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
            out.append(f"{k}={env.get(k)}: learner C's trunk (c0's, or the control stream) "
                       f"reads {k}=1 rows")
    if str(env.get("FRL_EXPERT_MODE", "")).strip() == "pinned" and not env.get("FRL_EXPERT"):
        out.append("FRL_EXPERT_MODE=pinned with no FRL_EXPERT: there are no expert rows to pin")
    # LANE-CENTRE SHAPING IS REMOVED FROM THIS ARM (user 2026-11-14: "Ok remove lanshaping then").
    # It is potential-based, so Q_shaped(s, a) = Q(s, a) - Phi(s): a function of the state alone,
    # which leaves dQ/da -- the only thing SAC's actor follows -- unchanged exactly. Job 466411 ran
    # it for 65k steps and metres per fault stayed at 56 / 55 (66 / 45 before) while strict success
    # fell 49/80 -> 43/80. Every learner-C overlay inherits 0 from slurm/v26_overlay.env.
    for k in ("LANE_CENTRE_K", "LANE_HEADING_K"):
        if float(env.get(k, "0") or 0.0) != 0.0:
            out.append(f"{k}={env[k]}: lane-centre shaping is removed from learner C (potential-"
                       f"based, so it cannot change dQ/da); set {k}=0")
    out += warm_start_refusals(env.get("FRL_CONTROL") or "", (env.get("FRL_ACTOR") or "frl").strip(),
                               int(env.get("FRL_IQL_PRETRAIN") or 0), env.get("FRL_EXPERT") or "")
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


class ControlTrunk:
    """The pretrained CONTROL STREAM (torch_rl.control_cnn.ControlGRU: a frame pair one decision
    apart -> small CNN -> GRU that also takes the speed; scripts/pretrain_control.py) as learner C's
    FROZEN trunk, with C0Trunk's interface: row -> {"embeddings": [stream GRU state | DEBiT
    binocular], "states": [mode, speed]} (user 2026-11-14).

    WHY IT WAS BUILT: c0's features carry lateral offset (R^2 0.70-0.81) but not heading error
    (0.046), and heading is what a steering action changes; the stream was meant to carry both.
    THE FIRST STREAM DOES NOT (corrected 2026-11-14): "heading 0.66-0.81" was a probe fitted and
    scored on the same rows. On 123 episodes control_pre_n2_k8/step_6000 never loaded, run as
    deployed, its own lane head gives offset R^2 0.38 and heading 0.10, and a probe with episodes
    held out 0.40 / 0.08 (scripts/control_stream_heldout.py). c0's GRU state, probed on the same
    episodes, gives 0.04 / 0.03 (scripts/control_stream_vs_c0.py): heading is in neither. The class
    is the plumbing for a stream that passes that gate; this one did not. DEBiT's binocular vector -- the
    goal stream -- is passed through untouched, as with c0.

    A SECOND MISMATCH, also measured 2026-11-14 and not removed here: pretraining never ran the GRU
    more than `chunk` steps from a zero state (12 for that checkpoint, 4 of them burn-in), while
    step() and embed_ticks() carry it for the whole episode; the state then drifts from the regime
    it was trained in (0.19 sd after decision 12, 0.56 sd after 64).

    IT IS FED WHAT PRETRAINING FED IT, each item pinned by tests/test_frl_control_trunk.py:
      * the frame pair is (the PREVIOUS DECISION's frame, this one), oldest first. At ACTION_REPEAT=k
        the env hands one frame per decision, k ticks apart, which is the stride the stream was
        pretrained at -- so the run's k must equal the stream's `repeat` (run_carla and load_policy
        refuse anything else). The first decision of an episode repeats its own frame, as
        control_data.stack_frames does at an episode start;
      * frames area-resized from the row's 128 px to the stream's px;
      * the speed in km/h: the row holds km/h / SPEED_SCALE_KMH and is multiplied back;
      * one GRU step per decision from a zero state at every episode start.
    ONE KNOWN DIFFERENCE FROM PRETRAINING, stated because it cannot be removed here: the speed's
    tick. scripts/pretrain_control.py paired frame t with the shard's speed[t], which is the speed
    AFTER tick t (the recorder stores it with frame t+1; torch_rl/expert_data.py), so pretraining
    read the speed one tick (0.1 s) AHEAD of its frame. The env can only give the speed at the
    frame, and embed_ticks takes the same (kmh[f] = the speed when frame f was taken): recorded
    and online rows agree with each other, and both differ from pretraining by that one tick.
    Measured 2026-11-14 on the same chunks under both pairings: the GRU state moved by 2.2% of its
    sd and an (in-sample) probe read the same either way, so the pairing itself changes little.
    The hidden is (GRU state, the previous decision's frame at px or None); the loops and the eval
    harness only reset it and pass it back.

    FROZEN, and that is a limit to state rather than a detail: the replay holds its features, so the
    critic trains the 50-d compression AFTER the stream but never the stream itself.
    """

    def __init__(self, ckpt_dir: str, device):
        from torch_rl import pixel_rows as pxr
        from torch_rl.control_cnn import ControlGRU
        if not (pxr.MONO_CHW and pxr.SPEED_OBS) or pxr.n_extra_state() != 1:
            raise ValueError("ControlTrunk reads [frame (3, s, s) | mode | speed | binoc] rows: "
                             "MONO_CHW=1 and SPEED_OBS=1, no PRIV_OBS")
        # tensors, strings and numbers only (scripts/pretrain_control.py), so no pickled code is run
        ck = torch.load(os.path.join(ckpt_dir, "params.pt"), map_location="cpu", weights_only=True)
        a = ck["args"]
        self.n_frames, self.px = int(a["n_frames"]), int(a["px"])
        self.repeat = int(a.get("repeat", 1))      # ticks per GRU step and inside the frame pair
        if self.n_frames not in (1, 2):
            raise ValueError(f"ControlTrunk carries ONE previous frame; this stream was pretrained "
                             f"with n_frames={self.n_frames}")
        self.device = torch.device(device)
        self.net = ControlGRU(latent_dim=int(a["latent"]), hidden_dim=int(a["hidden"]), px=self.px,
                              n_frames=self.n_frames).to(self.device)
        self.net.load_state_dict(ck["net"])        # strict: every module
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.c0_dir = ckpt_dir                     # the trunk's identity, as C0Trunk's (params_md5)
        self.hidden_dim = int(a["hidden"])
        self.binoc_dim = int(pxr.BINOC_DIM)
        self.state_dim = 2                         # [mode, speed / SPEED_SCALE_KMH], as c0's
        self.emb_dim = self.hidden_dim + self.binoc_dim
        self._img, self._pix = int(pxr.MONO_IMG), int(pxr.MONO_PIX)
        self._kmh = float(pxr.SPEED_SCALE_KMH)
        self._row = self._pix + self.state_dim + self.binoc_dim

    def zero_hidden(self):
        """(GRU state, the previous decision's frame): zeros and None -- a new episode has neither."""
        return (self.net.reset_hidden(1, self.device), None)

    def _small(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.px or x.shape[-2] != self.px:     # as scripts/pretrain_control.batch
            x = torch.nn.functional.interpolate(x, size=(self.px, self.px), mode="area")
        return x

    @torch.no_grad()
    def step(self, row, hidden):
        r = torch.as_tensor(np.asarray(row, np.float32), device=self.device).reshape(-1)
        if r.numel() != self._row:
            raise ValueError(f"ControlTrunk expects a [frame | mode | speed | binoc] row of "
                             f"{self._row} values, got {r.numel()}")
        P, I = self._pix, self._img
        cur = self._small(r[:P].reshape(1, 3, I, I))            # the row stores the frame / 255, CHW
        st, binoc = r[P:P + 2], r[P + 2:]
        h, prev = hidden
        x = cur if self.n_frames == 1 else torch.cat([cur if prev is None else prev, cur], dim=1)
        out, h = self.net(x, (st[1:2] * self._kmh).reshape(1, 1), h)      # the stream reads km/h
        obs = {"embeddings": torch.cat([out.float(), binoc[None]], dim=-1), "states": st[None]}
        return obs, (h, cur)

    @torch.no_grad()
    def embed_ticks(self, frames, binoc, mode, kmh, ticks):
        """The same features for RECORDED frames (the expert rows, torch_rl/frl/expert.py): one GRU
        step per tick in `ticks`, in order, from a zero state; each frame pair is
        control_data.stack_frames(frames, t, n_frames, repeat) -- what pretraining fed, including its
        clamp to frame 0 for a tick less than `repeat` into the episode. frames (N, 128, 128, 3)
        uint8 as the recorder stores them; kmh[f] = the speed when frame f was taken; binoc[f] its
        DEBiT vector. -> (embeddings (n, emb_dim), states (n, 2)), float32 numpy.

        For ticks 0, k, 2k, ... this equals step() on the rows the env would hand over for the same
        frames (pinned by test): recorded and online rows are ONE function of the frames."""
        from torch_rl.control_data import stack_frames
        if tuple(np.shape(frames)[1:]) != (self._img, self._img, 3):
            raise ValueError(f"recorded frames {np.shape(frames)} are not (N, {self._img}, "
                             f"{self._img}, 3): the env's rows hold {self._img} px frames")
        h = self.net.reset_hidden(1, self.device)
        embs, states = [], []
        for t in (int(t) for t in ticks):
            x = torch.from_numpy(stack_frames(frames, t, self.n_frames, self.repeat)[None])
            x = self._small(x.to(self.device).float() / 255.0)
            # the speed as a row carries it (float32 km/h / scale), multiplied back as step() does
            sp = torch.tensor([float(kmh[t]) / self._kmh], dtype=torch.float32, device=self.device)
            out, h = self.net(x, (sp * self._kmh).reshape(1, 1), h)
            b = torch.as_tensor(np.asarray(binoc[t], np.float32).reshape(-1), device=self.device)
            embs.append(torch.cat([out[0].float(), b]))
            states.append(torch.cat([torch.tensor([float(mode)], device=self.device), sp]))
        if not embs:
            return (np.zeros((0, self.emb_dim), np.float32),
                    np.zeros((0, self.state_dim), np.float32))
        return torch.stack(embs).cpu().numpy(), torch.stack(states).cpu().numpy()


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
                 actor: str = "frl", trunk: Optional[C0Trunk] = None, batched_heads: bool = False):
    """Learner C. actor="c0": its actor is replaced by a trainable copy of c0's head (C0HeadActor).
    batched_heads: the critic's heads in one batched pass (the same weights and numbers, faster)."""
    from torch_rl.frl.learner import SACLearner
    learner = SACLearner(action_dim=action_dim, embedding_dim=emb_dim, state_dim=state_dim,
                         latent_dim=LATENT_DIM, seed=seed, device=str(device),
                         batched_heads=bool(batched_heads), **REDQ)
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
    train_carla_sac.run_eval drives it unchanged. The hidden is the trunk's: c0's GRU state, or
    the control stream's (GRU state, previous frame) -- opaque to the harness either way."""

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
    # THE TRUNK IS PART OF THE POLICY: the actor was trained on THIS trunk's features -- c0's, or
    # the control stream's when the checkpoint's settings name one. FRL_C0 / FRL_CONTROL may only
    # point at the same trunk elsewhere (e.g. another machine's copy), checked by md5.
    control = bool((meta.get("settings") or {}).get("control"))
    kind = "control stream" if control else "c0 trunk"
    c0 = os.environ.get("FRL_CONTROL" if control else "FRL_C0") or meta["c0"]
    if params_md5(c0) != meta["c0_md5"]:
        raise ValueError(f"{kind} {c0} (md5 {params_md5(c0)}) is not the one this checkpoint "
                         f"was trained on ({meta['c0']}, md5 {meta['c0_md5']})")
    trunk = ControlTrunk(c0, device) if control else C0Trunk(c0, device)
    if control:
        # A policy trained at one decision per k ticks, driven at another rate, would be handed
        # frame pairs at the wrong spacing and nothing would look wrong -- so it is refused.
        from torch_rl.carla_env_build import action_repeat
        if trunk.repeat != action_repeat():
            raise ValueError(f"{d} was trained on a control stream that decides once per "
                             f"{trunk.repeat} ticks; this env acts at ACTION_REPEAT="
                             f"{action_repeat()}")
    if (trunk.emb_dim, trunk.state_dim) != (meta["emb_dim"], meta["state_dim"]):
        raise ValueError(f"{kind} dims {(trunk.emb_dim, trunk.state_dim)} != the "
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


def _iql_warm_up(learner, expert, s: LoopSettings, log: Callable, resumed: bool = False) -> None:
    """LoopSettings.iql_pretrain: IQL on the expert rows BEFORE the first env step (user 2026-11-14:
    "also pretrain critic with IQL first"). torch_rl.frl.iql.iql_warmup applies FastRLAP's IQL rules
    to the learner's OWN critic and actor, so the SAC loop simply continues with them; the
    temperature is not touched (recorded rows never tune it). A resumed learner is already past it.
    With no expert rows it is an error, never a silent skip."""
    if not s.iql_pretrain:
        return
    if resumed:
        log(f"[IQL] warm-up skipped: the resumed learner is past it (iql_pretrain="
            f"{s.iql_pretrain})")
        return
    if expert is None:
        raise ValueError(f"iql_pretrain={s.iql_pretrain} needs an expert replay (FRL_EXPERT): the "
                         f"warm-up learns from the expert rows")
    from torch_rl.frl.iql import iql_warmup
    t0 = time.time()
    log(f"[IQL] warm-up: {s.iql_pretrain} updates on batches of {s.batch} of the {len(expert)} "
        f"expert rows, before the first env step")
    info = iql_warmup(learner, lambda: expert.sample(s.batch), s.iql_pretrain, log=log)
    log(f"[IQL] warm-up done in {time.time() - t0:.0f} s"
        + "".join(f" {k}={v:.4g}" for k, v in sorted((info or {}).items())
                  if isinstance(v, (int, float))))


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
                               actor=s.actor, trunk=trunk, batched_heads=bool(s.batched_heads))
    pinned = expert is not None and s.expert_mode == "pinned"
    # pinned: the expert rows sit in THIS replay (never overwritten) and there is no second update
    rb = FeatureReplay(s.replay, trunk.emb_dim, trunk.state_dim, act_dim,
                       s.replay_device or s.device, s.seed, pinned=expert.t if pinned else None)
    _iql_warm_up(learner, expert, s, log)        # before the seeds below: the loop's streams unchanged
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


# Settings a resumed run may change: run length and cadence, devices, the loop, the resume itself (and
# the seed, whose streams cannot be continued anyway). Every other LoopSettings field must match.
RESUME_MAY_DIFFER = ("max_steps", "ckpt_every", "log_every", "device", "replay_device", "parallel",
                     "resume", "seed", "expert", "batched_heads", "resume_new_actor",
                     # the stream's PATH (its identity is the checkpoint's c0_md5, checked below) and
                     # the warm-up, which a resumed learner is past
                     "control", "iql_pretrain")


def restore_learner(learner, ckpt_dir: str, s: LoopSettings, c0_dir: str) -> int:
    """Load a learner-C checkpoint's whole learner (actor, critic, target critic, temperature, entropy
    target, the three optimizers) into `learner`; returns its step. Refused (ValueError) when it was
    made on another c0 trunk, another expert file, other networks or other learning settings.

    With s.resume_new_actor the ACTOR AND ITS OPTIMIZER ARE NOT LOADED: `learner` keeps the actor it was
    built with (actor="c0" -> a trainable copy of c0's head) on the checkpoint's trained critic, and
    `actor` is allowed to differ from the checkpoint's."""
    with open(os.path.join(ckpt_dir, "config.json")) as fh:
        meta = json.load(fh)
    if meta.get("learner") != "frl":
        raise ValueError(f"{ckpt_dir} is not a learner-C checkpoint")
    if meta["c0_md5"] != params_md5(c0_dir):
        raise ValueError(f"{ckpt_dir} was trained on trunk md5 {meta['c0_md5']} (c0's, or the "
                         f"control stream's); this trunk is {params_md5(c0_dir)}")
    redq = {k: list(v) if isinstance(v, tuple) else v for k, v in REDQ.items()}
    if meta.get("redq") != redq or meta.get("latent_dim") != LATENT_DIM:
        raise ValueError(f"{ckpt_dir}: networks {meta.get('redq')} latent {meta.get('latent_dim')} "
                         f"!= this run's {redq} latent {LATENT_DIM}")
    old, new = meta["settings"], dataclasses.asdict(s)
    new_actor = bool(int(getattr(s, "resume_new_actor", 0) or 0))
    # The whole point of resume_new_actor is a DIFFERENT actor on the checkpoint's critic.
    may_differ = set(RESUME_MAY_DIFFER) | ({"actor"} if new_actor else set())
    bad = [f"{k}: checkpoint {old.get(k)!r} vs run {new[k]!r}" for k in new
           if k not in may_differ and old.get(k, new[k]) != new[k]]
    if os.path.basename(old.get("expert") or "") != os.path.basename(new.get("expert") or ""):
        bad.append(f"expert: checkpoint {old.get('expert')!r} vs run {new.get('expert')!r}")
    if bad:
        raise ValueError(f"{ckpt_dir} was made with other learning settings -- " + "; ".join(bad))
    st = torch.load(os.path.join(ckpt_dir, "params.pt"), map_location=str(learner.device))
    if not new_actor:
        learner.actor.load_state_dict(st["actor"])
        learner.actor_opt.load_state_dict(st["actor_opt"])
    learner.critic.load_state_dict(st["critic"])
    learner.target_critic.load_state_dict(st["target_critic"])
    with torch.no_grad():
        learner.log_temp.copy_(st["log_temp"].to(learner.log_temp.device))
    learner.target_entropy = float(st["target_entropy"])
    learner.critic_opt.load_state_dict(st["critic_opt"])
    learner.temp_opt.load_state_dict(st["temp_opt"])
    return int(meta["step"])


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
                               actor=s.actor, trunk=trunk, batched_heads=bool(s.batched_heads))
    pinned = expert is not None and s.expert_mode == "pinned"
    # RESUME: the whole learner and the step count from a checkpoint; the replay is not in it, so it starts
    # fresh, and the random streams are reseeded past the resumed step (they cannot be continued).
    start = restore_learner(learner, s.resume, s, c0_dir) if s.resume else 0
    if s.resume:
        log(f"[Resume] {s.resume}: learner restored at step {start}; fresh replay (pinned expert rows "
            f"{'+ ' if pinned else ''}own rows from 0); updates resume when {s.start_training} own rows "
            f"are back")
    rb = FeatureReplay(s.replay, trunk.emb_dim, trunk.state_dim, act_dim,
                       s.replay_device or s.device, s.seed + start,
                       pinned=expert.t if pinned else None)
    _iql_warm_up(learner, expert, s, log, resumed=bool(s.resume))
    rng = np.random.default_rng(s.seed + start)
    torch.manual_seed(s.seed + start)
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
        step, last_ckpt = start, start
        next_log = (start // s.log_every + 1) * s.log_every
        next_ckpt = (start // s.ckpt_every + 1) * s.ckpt_every
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
                # a resumed run first refills its own rows to start_training (its fresh replay)
                if step >= s.start_training and (not s.resume or rb.n >= s.start_training):
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
                    f"{(step - start) / max(1e-9, time.time() - t0):.1f} steps/s"
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
    bad = warm_start_refusals(s.control, s.actor, s.iql_pretrain, s.expert)
    if bad:
        raise SystemExit("LEARNER=frl refuses these settings -- " + "; ".join(bad))
    cache = os.environ.get("FRL_PROBE_CACHE", "")
    if s.control:
        from torch_rl.carla_env_build import action_repeat as _repeat
        trunk = ControlTrunk(s.control, s.device)
        c0_dir = s.control            # the trunk's identity from here on: md5, checkpoints, resume
        if trunk.repeat != _repeat():
            raise SystemExit(f"FRL_CONTROL {s.control} was pretrained at one decision per "
                             f"{trunk.repeat} ticks; this run acts at ACTION_REPEAT={_repeat()}")
        cache = ""                    # the steering probe's cache holds c0's features
    else:
        c0_dir = os.environ.get("FRL_C0", "")
        if not c0_dir:
            raise SystemExit("LEARNER=frl needs FRL_C0: c0's checkpoint directory (frozen trunk)")
        trunk = C0Trunk(c0_dir, s.device)
    os.makedirs(out, exist_ok=True)
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
            raise SystemExit(f"FRL_EXPERT {s.expert} was built on trunk md5 {expert.c0_md5}; this "
                             f"run's trunk ({'the control stream' if s.control else 'c0'}) is "
                             f"{params_md5(c0_dir)}")
    ramp = make_ramp()
    if ramp is not None:
        obs, info = env.reset()       # main drew the first goal at the target fraction
    e = os.environ.get
    log(f"=== [{e('RUN_NAME')}] preflight: LEARNER C (FastRLAP SACLearner, embeddings mode) ===")
    front = (f"the CONTROL STREAM (pretrained ControlGRU: {trunk.n_frames} frame(s) {trunk.repeat} "
             f"ticks apart at {trunk.px} px, one GRU step per decision), FROZEN," if s.control
             else "c0 FROZEN trunk")
    log(f"  front end: {front} {c0_dir} -> embeddings [GRU {trunk.hidden_dim} | binoc "
        f"{trunk.binoc_dim}] = {trunk.emb_dim}-d, states [mode, speed] = {trunk.state_dim}-d; "
        f"each -> Dense({LATENT_DIM}) -> LayerNorm -> tanh (critic's trained by the critic)")
    log(f"  learner: {REDQ} latent {LATENT_DIM}; "
        + ("actor and critics FROM THE IQL WARM-UP, temperature from scratch" if s.iql_pretrain
           else "actor/critics/temperature FROM SCRATCH"))
    log(f"  IQL WARM-UP: {s.iql_pretrain} updates on the expert rows before the first env step "
        f"(FastRLAP's IQL rules on this learner's own critic and actor, with its own optimizers; "
        f"the temperature untouched; skipped on resume)" if s.iql_pretrain
        else "  IQL WARM-UP: off")
    log(f"  loop: {dataclasses.asdict(s)}; random actions for steps < {s.start_training}; one "
        f"update of {s.batch} split into {s.utd} minibatches per step; mask 0 only on "
        f"termination. NOT used: HER, noise-free episodes, junction curriculum, demos"
        + ("" if s.iql_pretrain else ", warm start"))
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
    log(f"  {'control stream' if s.control else 'c0 trunk'} md5 {params_md5(c0_dir)} | the gated "
        f"loop's settings are NOT read: "
        + " ".join(f"{k}={e(k)}" for k in ("MAX_STEPS", "CKPT_EVERY", "LEARNING_STARTS",
                                           "UPDATE_EVERY", "REPLAY_CAPACITY") if e(k)))
    log(f"  env rebuild on a failed reset: {'on' if rebuild_env else 'OFF'}")
    own = "learner C (from the IQL warm-up)" if s.iql_pretrain else "learner C (from scratch)"
    log(f"  ACTOR: {'c0 head (trainable copy; the RL policy starts as c0)' if s.actor == 'c0' else own}"
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
