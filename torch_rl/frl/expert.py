"""Expert prior data for learner C (user 2026-11-10: "can we use c0 expert data locally with run like fastrlap
but rl collects data like now").

FastRLAP's online loop (fastrlap-mujoco-env/train_online_pixels.py) keeps a second, fixed replay of prior data
and, every step, makes one more `agent.update(batch_expert, utd_ratio, update_temperature=False)` after the
update on its own data. Here the prior data are c0's own expert demonstrations (the M1 shards it was cloned
from), turned into learner C's transitions exactly as the online loop (torch_rl/frl/carla.py) sees the world:
  * features from c0's frozen trunk stepped ONCE PER DECISION -- every k-th frame at action repeat k -- from a
    zero GRU state at the phase's first frame (phase 0 = the episode start; phases 1..k-1 start one tick later
    each, giving every recorded tick a decision of its own);
  * a decision's reward = the sum of its k ticks; its next observation = the frame k ticks on; an episode end
    inside the window ends it early, as torch_rl.carla_env_build.ActionRepeat stops early; its action = the
    action executed at the decision's first tick;
  * rewards RELABELLED to the run's reward: the idle charge added (EXPLORE_IDLE_PENALTY below
    EXPLORE_IDLE_SPEED_KMH, both modes; M1 was recorded without it) and the explore junction bonus removed
    (recorded at 10, off in the run: it paid circling, memory explore-junction-bonus-farmable);
  * the no_arrival cut of the offline study: a reached episode loses its last ARRIVAL_CUT ticks and ends
    truncated, so the critic is never paid the +100 for the expert's approach;
  * expert-driven episodes only; the steering probe's held-out episodes excluded.
The replay is fixed (never overwritten) and never moves the temperature -- the prerecorded-data rule.
"""
import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from torch_rl.slope_diag import ARRIVAL_CUT


def relabel_rewards(reward, speed, mode: int, idle_penalty: float, idle_kmh: float,
                    junction_bonus: float) -> np.ndarray:
    """Recorded per-tick rewards -> the run's: in explore mode (0) every tick paying more than half the
    junction bonus loses one bonus; every tick below `idle_kmh` pays the idle charge."""
    out = np.asarray(reward, np.float32).copy()
    if junction_bonus and int(mode) == 0:
        out[out > junction_bonus / 2.0] -= junction_bonus
    if idle_penalty:
        out[np.asarray(speed, np.float32) < idle_kmh] -= idle_penalty
    return out


def decision_windows(T: int, k: int, phase: int) -> List[tuple]:
    """(first tick, end tick) of each decision of one phase over T ticks; the last may be short."""
    return [(f, min(f + k, T)) for f in range(int(phase), int(T), int(k))]


def select_episodes(names: Sequence[str], drivers: Sequence[str],
                    held_out: Optional[set] = None) -> List[str]:
    """Expert-driven episodes, minus the probe's held-out ones: `held_out` names, or by default the probe's
    rule (scripts/offline_priv_critic.held_out_split) -- the first 3 and every 10th of the sorted names."""
    if held_out is None:
        order = sorted(names)
        held_out = set(order[:3]) | set(order[::10])
    return [n for n, d in zip(names, drivers) if str(d) == "expert" and n not in held_out]


@torch.no_grad()
def episode_transitions(trunk, ep: Dict, k: int, idle_penalty: float, junction_bonus: float,
                        idle_kmh: float = 2.0, cut: int = ARRIVAL_CUT) -> Dict[str, np.ndarray]:
    """One recorded episode -> learner C transitions at action repeat k (every phase)."""
    from torch_rl import pixel_rows as pxr
    T = int(len(ep["act_exec"]))
    mode = int(np.asarray(ep["mode"]))
    if str(np.asarray(ep["end"])) == "reached":
        if T <= cut:
            return _empty(trunk)
        T_eff, terminal = T - cut, False
    else:
        T_eff, terminal = T, bool(np.asarray(ep["terminal"]))
    rew = relabel_rewards(ep["reward"][:T], ep["speed"][:T], mode, idle_penalty, idle_kmh,
                          junction_bonus)
    speed = np.asarray(ep["speed"], np.float32)
    kmh = np.concatenate([[0.0], speed])                   # frame f pairs with the speed after tick f-1
    rows = np.stack([pxr.mono_row(ep["frames"][f], float(mode), np.asarray(ep["binoc"][f], np.float32),
                                  speed_kmh=float(kmh[f])) for f in range(T_eff + 1)])
    a = trunk.agent
    x = torch.as_tensor(rows, device=trunk.device)
    e = a._embed(a.compress, a.act_embed, x).float()
    st = a._state(x).float()
    out = {key: [] for key in ("emb", "states", "next_emb", "next_states", "actions", "rewards",
                               "masks")}
    for phase in range(int(k)):
        wins = decision_windows(T_eff, k, phase)
        if not wins:
            continue
        visit = [f for f, _ in wins] + [wins[-1][1]]
        hs, _ = a.core(e[visit][None], torch.zeros(1, 1, trunk.hidden_dim, device=trunk.device))
        hs = hs[0].float()
        sv = st[visit]
        emb = torch.cat([hs, sv[:, trunk.state_dim:]], dim=-1).cpu().numpy()
        states = sv[:, :trunk.state_dim].cpu().numpy()
        for j, (f, end) in enumerate(wins):
            out["emb"].append(emb[j])
            out["states"].append(states[j])
            out["next_emb"].append(emb[j + 1])
            out["next_states"].append(states[j + 1])
            out["actions"].append(np.asarray(ep["act_exec"][f], np.float32))
            out["rewards"].append(float(rew[f:end].sum()))
            out["masks"].append(0.0 if (terminal and end == T_eff) else 1.0)
    return {"emb": np.asarray(out["emb"], np.float16),
            "next_emb": np.asarray(out["next_emb"], np.float16),
            "states": np.asarray(out["states"], np.float32),
            "next_states": np.asarray(out["next_states"], np.float32),
            "actions": np.asarray(out["actions"], np.float32),
            "rewards": np.asarray(out["rewards"], np.float32),
            "masks": np.asarray(out["masks"], np.float32)}


def _empty(trunk) -> Dict[str, np.ndarray]:
    return {"emb": np.zeros((0, trunk.emb_dim), np.float16),
            "next_emb": np.zeros((0, trunk.emb_dim), np.float16),
            "states": np.zeros((0, trunk.state_dim), np.float32),
            "next_states": np.zeros((0, trunk.state_dim), np.float32),
            "actions": np.zeros((0, 2), np.float32), "rewards": np.zeros(0, np.float32),
            "masks": np.zeros(0, np.float32)}


def concat(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def save_replay(path: str, tr: Dict[str, np.ndarray], *, k: int, c0_md5: str, meta: Dict) -> str:
    np.savez(path, **tr, k=np.array(int(k)), c0_md5=np.array(str(c0_md5)),
             meta=np.array(json.dumps(meta)))
    return path


class ExpertReplay:
    """The fixed expert replay (CPU), sampled uniformly with replacement like FeatureReplay."""

    def __init__(self, arrays: Dict[str, np.ndarray], k: int, c0_md5: str, meta: Dict, seed: int = 0):
        self.t = {key: torch.from_numpy(np.ascontiguousarray(v)) for key, v in arrays.items()}
        self.k, self.c0_md5, self.meta = int(k), str(c0_md5), meta
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "ExpertReplay":
        z = np.load(path, allow_pickle=False)
        arrays = {key: z[key] for key in ("emb", "states", "next_emb", "next_states", "actions",
                                          "rewards", "masks")}
        return cls(arrays, int(z["k"]), str(z["c0_md5"]), json.loads(str(z["meta"])), seed)

    def __len__(self) -> int:
        return int(self.t["rewards"].shape[0])

    def sample(self, batch: int) -> Dict:
        i = torch.as_tensor(self.rng.integers(0, len(self), size=int(batch)))
        t = self.t
        return {"observations": {"embeddings": t["emb"][i].float(), "states": t["states"][i]},
                "next_observations": {"embeddings": t["next_emb"][i].float(),
                                      "states": t["next_states"][i]},
                "actions": t["actions"][i], "rewards": t["rewards"][i], "masks": t["masks"][i]}


def expert_update(learner, rb: ExpertReplay, batch: int, utd: int) -> Dict:
    """FastRLAP's second update of a step: an expert batch, the temperature left alone."""
    return learner.update(rb.sample(batch), utd_ratio=utd, update_temperature=False)
