#!/usr/bin/env python3
"""Episode replay for a recurrent learner (spec section 3, choice C1).

Chunks are sampled WITHIN one episode, which is what lets the learner roll a
cuDNN GRU over the whole chunk: there is no mid-chunk hidden reset to model.
Latents are 3136 floats per step, so they are stored fp16: 1M steps is 6.3 GB
instead of 12.5 GB.
"""
from typing import Dict, List

import numpy as np
import torch


class _Episode:
    __slots__ = ("obs", "actions", "rewards", "masks")

    def __init__(self, obs_dim: int, action_dim: int):
        self.obs: List[np.ndarray] = []          # length n + 1
        self.actions: List[np.ndarray] = []      # length n
        self.rewards: List[float] = []
        self.masks: List[float] = []

    def __len__(self) -> int:
        return len(self.actions)


class EpisodeBuffer:
    obs_dtype = np.float16

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: int = 0) -> None:
        self.obs_dim, self.action_dim, self.capacity = int(obs_dim), int(action_dim), int(capacity)
        self.rng = np.random.default_rng(seed)
        self._episodes: List[_Episode] = []
        self._open = _Episode(self.obs_dim, self.action_dim)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def episodes(self) -> int:
        return len(self._episodes)

    @property
    def sampleable(self) -> int:
        """Number of transitions that can actually be sampled.

        This counts transitions only in ENDED episodes. The `size` property includes
        all transitions (ended + open), but the sampler only draws from ended episodes,
        which ensures no chunk spans two episodes. A training loop must gate on
        `sampleable >= threshold` before calling `sample_chunks`.
        """
        return sum(len(e) for e in self._episodes)

    def add(self, obs, action, reward: float, terminal: bool, truncated: bool, next_obs) -> None:
        if not self._open.obs:
            self._open.obs.append(np.asarray(obs, self.obs_dtype))
        self._open.actions.append(np.asarray(action, np.float32))
        self._open.rewards.append(float(reward))
        # A time-limit cut still bootstraps; only a real terminal stops the backup.
        self._open.masks.append(0.0 if terminal else 1.0)
        self._open.obs.append(np.asarray(next_obs, self.obs_dtype))
        self._size += 1
        if terminal or truncated:
            self.end_episode()

    def end_episode(self) -> None:
        if len(self._open) > 0:
            self._episodes.append(self._open)
        self._open = _Episode(self.obs_dim, self.action_dim)
        while self._size > self.capacity and len(self._episodes) > 1:
            self._size -= len(self._episodes.pop(0))

    def sample_chunks(self, n_chunks: int, length: int) -> Dict[str, torch.Tensor]:
        if not self._episodes:
            raise ValueError(
                f"EpisodeBuffer has no ended episodes to sample (size {self._size}, sampleable {self.sampleable}); "
                f"gate on .sampleable"
            )
        L, B = int(length), int(n_chunks)
        obs = np.zeros((B, L + 1, self.obs_dim), np.float32)
        actions = np.zeros((B, L, self.action_dim), np.float32)
        rewards = np.zeros((B, L), np.float32)
        masks = np.zeros((B, L), np.float32)
        valid = np.zeros((B, L), np.float32)
        weights = np.array([len(e) for e in self._episodes], np.float64)
        weights /= weights.sum()
        for b in range(B):
            ep = self._episodes[self.rng.choice(len(self._episodes), p=weights)]
            n = len(ep)
            start = 0 if n <= L else int(self.rng.integers(0, n - L + 1))
            take = min(L, n - start)
            obs[b, :take + 1] = np.asarray(ep.obs[start:start + take + 1], np.float32)
            actions[b, :take] = np.asarray(ep.actions[start:start + take], np.float32)
            rewards[b, :take] = np.asarray(ep.rewards[start:start + take], np.float32)
            masks[b, :take] = np.asarray(ep.masks[start:start + take], np.float32)
            valid[b, :take] = 1.0
        return {"obs": torch.from_numpy(obs), "actions": torch.from_numpy(actions),
                "rewards": torch.from_numpy(rewards), "masks": torch.from_numpy(masks),
                "valid": torch.from_numpy(valid)}


class _OpenEpisode:
    """One env's staged, not-yet-ended episode (DeviceEpisodeBuffer).

    `obs` rows are either all `np.ndarray` (host-staged, as Phase 1) or all
    `torch.Tensor` (staged on whatever device they arrive on -- Vc1VecEnv
    hands back CUDA rows on the learner device); `is_torch` records which, so
    `add()` can reject a mid-episode mix and `end_episode()` knows whether to
    build the row block with `np.asarray` (host) or `torch.stack` (no host
    round trip). `actions`/`rewards`/`masks` stay host scalars/arrays either
    way, as today. `goal` is the episode's (goal_dim,) fp16 row on the buffer's
    device (None when the buffer has goal_dim == 0), validated at the opening
    add() and stored at end_episode().
    """
    __slots__ = ("obs", "actions", "rewards", "masks", "mode", "is_torch", "goal")

    def __init__(self, mode: int, is_torch: bool, goal=None) -> None:
        self.obs: List = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.masks: List[float] = []
        self.mode = mode
        self.is_torch = is_torch
        self.goal = goal


class DeviceEpisodeBuffer:
    """Episode replay staged on host, stored on `device`, sampled with uniform row coverage.

    `EpisodeBuffer` draws a chunk's START uniformly in [0, n - L]. That trains
    a middle row ~(L - K)x more often than the episode's LAST transition (the
    one carrying the arrival bonus), and never trains using the first K rows
    as burn-in context. Here the TRAINED WINDOW start t0 is drawn uniformly in
    [0, n - 1] instead of the chunk start, so every transition -- including
    the terminal one -- is the trained target of the same number of draws
    (T = length - burn_in of them; see the coverage test).

    Storage lives on `device` as one flat array of rows -- `capacity` counts
    ROWS, not transitions. An episode of n transitions occupies rows
    [start, start + n]: row start+i holds obs_i and, for i < n, also
    action_i/reward_i/mask_i; the last row holds only obs_n (no
    action/reward/mask -- it is never returned as a valid transition). The
    open episode is staged on the host as plain lists and copied to `device`
    in a single transfer per episode at `end_episode()`; nothing per-step
    touches the device.

    Episodes are tagged goal (mode=1) or explore (mode=0) at add() time, and
    `sample_chunks` stratifies a batch by mode (`goal_batch_frac`) before
    picking an episode (probability proportional to its length) and a window
    start within it.

    `add`/`end_episode` take an `env_id` (default 0), one open (host-staged)
    episode per env_id, so N parallel envs can interleave `add()` calls in any
    order without their steps mixing into the same episode: the mode-mismatch
    check and the atomic-failure semantics below are scoped to a single
    env_id's open episode, never across env_ids. `_episodes` (the ended,
    sampleable pool) and everything derived from it -- `sampleable`,
    `sampleable_by_mode`, `size`, `episodes`, `sample_chunks` -- are common
    across all envs, exactly as if the episodes had arrived from one env
    serially: interleaving changes only which rows land in the ring at what
    time, never what a finished episode contains.

    `goal_dim > 0` adds ONE goal row per episode (an image goal's tokens, say):
    the opening `add()` of each episode must hand `goal=`, it is stored fp16 on
    `device` in `_goals` keyed by the episode's ring start row (unique per live
    episode; the `_episodes` index stays a list of (start, n, mode) 3-tuples),
    dropped with the episode's rows on eviction, and `sample_chunks` returns it
    per chunk as `"goal": (B, goal_dim)` float32. With `goal_dim == 0` (the
    default) none of this exists and the buffer is exactly the goal-free one.
    """

    obs_dtype = np.float16

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device,
                 seed: int = 0, goal_batch_frac: float = 0.5, goal_dim: int = 0,
                 store_device=None, u8_cols: int = 0) -> None:
        """`device` is where SAMPLED batches are delivered (the learner's).
        `store_device` is where the ring lives; default `device`. Set it to
        "cpu" to hold the replay in host RAM -- user 2026-10-19, "move to main
        memory": the L40 fits ~350k rows in fp16, host RAM fits millions.

        `u8_cols` (default 0 = off): the first `u8_cols` obs columns are
        stored as uint8 levels round(x * 255) and returned as level / 255. Only
        for columns that ARE k/255 for integer k (the /255 camera frame of
        pixel_rows.mono_row), for which the round trip is EXACT; it halves
        those columns' memory. Anything else in those columns is refused."""
        self.obs_dim, self.action_dim, self.capacity = int(obs_dim), int(action_dim), int(capacity)
        self.goal_dim = int(goal_dim)
        self.device = torch.device(device)
        self.store_device = torch.device(store_device) if store_device is not None else self.device
        self.u8_cols = int(u8_cols)
        if not 0 <= self.u8_cols <= self.obs_dim:
            raise ValueError(f"u8_cols={u8_cols} outside [0, obs_dim={self.obs_dim}]")
        self.goal_batch_frac = float(goal_batch_frac)
        self.rng = np.random.default_rng(seed)

        sd, U = self.store_device, self.u8_cols
        self.obs_u8 = torch.zeros((self.capacity, U), dtype=torch.uint8, device=sd)
        self.obs = torch.zeros((self.capacity, self.obs_dim - U), dtype=torch.float16, device=sd)
        self.actions = torch.zeros((self.capacity, self.action_dim), dtype=torch.float32, device=sd)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=sd)
        self.masks = torch.zeros((self.capacity,), dtype=torch.float32, device=sd)

        self._episodes: List = []   # ended episodes, FIFO (oldest first): (start, n, mode)
        self._write = 0             # next free row
        self._size = 0              # transitions in ended + open episodes (mirrors EpisodeBuffer.size)
        self._open: Dict[int, _OpenEpisode] = {}   # env_id -> its one open episode
        self._goals: Dict[int, torch.Tensor] = {}  # episode start row -> (goal_dim,) fp16 on store_device

    @property
    def nbytes(self) -> int:
        """Bytes held by the ring (not counting the per-episode goal rows)."""
        return sum(t.numel() * t.element_size() for t in
                   (self.obs_u8, self.obs, self.actions, self.rewards, self.masks))

    def _put_obs(self, start: int, rows: torch.Tensor) -> None:
        """Write (k, obs_dim) rows: the u8 columns as exact levels, the rest fp16."""
        U = self.u8_cols
        if U:
            pix = rows[:, :U].to(torch.float32) * 255.0
            lv = torch.round(pix)
            if bool((lv < 0).any()) or bool((lv > 255).any()) or \
                    bool(((pix - lv).abs() > 0.25).any()):  # levels are 1 apart; fp16 input rounds by <=0.07
                raise ValueError(
                    f"u8_cols={U}: columns are not k/255 pixel levels, so storing "
                    "them as uint8 would change them")
            self.obs_u8[start:start + rows.shape[0]] = lv.to(
                device=self.store_device, dtype=torch.uint8)
        self.obs[start:start + rows.shape[0]] = rows[:, U:].to(
            device=self.store_device, dtype=torch.float16)

    @property
    def size(self) -> int:
        return self._size

    @property
    def episodes(self) -> int:
        return len(self._episodes)

    @property
    def sampleable(self) -> int:
        """Transitions in ENDED episodes only -- what `sample_chunks` can actually draw from."""
        return sum(n for _, n, _ in self._episodes)

    @property
    def sampleable_by_mode(self) -> Dict[int, int]:
        out = {0: 0, 1: 0}
        for _, n, mode in self._episodes:
            out[mode] += n
        return out

    def add(self, obs, action, reward: float, terminal: bool, truncated: bool, next_obs, mode,
            env_id: int = 0, goal=None) -> None:
        """`obs`/`next_obs` may each be a numpy array OR a torch tensor (on any
        device -- a vector env passes one row of an on-device CUDA batch, e.g.
        `rows[i]`); an open episode must be all-numpy or all-torch, never
        mixed. `action` stays a numpy array (or array-like), as today.

        `goal` (buffers with goal_dim > 0 only): the episode's goal, a 1-d
        numpy array or torch tensor (any device/dtype) of goal_dim values.
        Required on the OPENING add() of an episode, ignored on later ones."""
        m = int(mode)
        if m not in (0, 1):
            raise ValueError(f"mode must be 0 (explore) or 1 (goal), got {mode!r}")
        obs_is_torch = isinstance(obs, torch.Tensor)
        next_is_torch = isinstance(next_obs, torch.Tensor)
        ep = self._open.get(env_id)
        # Every check below runs BEFORE any mutation (creating/registering a
        # new _OpenEpisode, or appending to an existing one): a call that
        # fails any of them must leave self._open/self._size/every episode
        # exactly as they were, including on the OPENING call, where there is
        # no pre-existing `ep` yet to leave untouched -- a rejected opening
        # call must not register one either, or its single (already-appended)
        # obs row survives as a stray leading row of whatever episode opens
        # next on this env_id (round 1's bug).
        if ep is not None and m != ep.mode:
            raise ValueError(
                f"mode changed mid-episode (env_id={env_id}): episode opened as mode "
                f"{ep.mode}, add() got {m}"
            )
        expected_is_torch = ep.is_torch if ep is not None else obs_is_torch
        if obs_is_torch != expected_is_torch or next_is_torch != expected_is_torch:
            raise ValueError(
                f"episode rows must be all-numpy or all-torch, never mixed (env_id={env_id}): "
                f"episode is {'torch' if expected_is_torch else 'numpy'}, this add() got obs="
                f"{'torch' if obs_is_torch else 'numpy'} next_obs="
                f"{'torch' if next_is_torch else 'numpy'}"
            )
        goal_t = None
        if ep is None and self.goal_dim > 0:
            # Validated on the opening call, before anything is registered: a
            # missing or malformed goal must leave self._open untouched, like
            # every other rejected opening call above.
            if goal is None:
                raise ValueError(
                    f"goal_dim={self.goal_dim}: the opening add() of an episode needs goal="
                )
            goal_t = goal.detach() if isinstance(goal, torch.Tensor) else torch.as_tensor(np.asarray(goal))
            goal_t = goal_t.reshape(-1)
            if goal_t.numel() != self.goal_dim:
                raise ValueError(
                    f"goal has {goal_t.numel()} values, expected goal_dim={self.goal_dim}"
                )
            # copy=True: the stored row never aliases the caller's array/tensor
            # (a row of an env-owned batch that is overwritten next step, say).
            goal_t = goal_t.to(device=self.store_device, dtype=torch.float16, copy=True)
        if ep is None:
            ep = _OpenEpisode(m, is_torch=obs_is_torch, goal=goal_t)
            # Torch rows are kept as-is (no dtype/device cast here): a per-step
            # cast on a CUDA row would itself be the kind of extra device op
            # this fix avoids. Numpy rows cast to obs_dtype now, as today --
            # cheap, host-side, and unrelated to the device-copy issue.
            ep.obs.append(obs if obs_is_torch else np.asarray(obs, self.obs_dtype))
            self._open[env_id] = ep
        ep.actions.append(np.asarray(action, np.float32))
        ep.rewards.append(float(reward))
        # A time-limit cut still bootstraps; only a real terminal stops the backup.
        ep.masks.append(0.0 if terminal else 1.0)
        ep.obs.append(next_obs if next_is_torch else np.asarray(next_obs, self.obs_dtype))
        self._size += 1
        if terminal or truncated:
            self.end_episode(env_id)

    def end_episode(self, env_id: int = 0) -> None:
        ep = self._open.pop(env_id, None)
        if ep is None or len(ep.actions) == 0:
            return
        n = len(ep.actions)
        try:
            start = self._reserve_rows(n)
        except ValueError:
            # Atomic failure, scoped to this env_id: `add()` already counted these n
            # transitions into `_size`; `ep` is already popped out of `self._open` above
            # (before `_reserve_rows` can raise), so undoing `_size` leaves the buffer
            # exactly as it was before this episode's first add() on this env_id --
            # a following add(env_id=env_id) in either mode starts a fresh episode
            # rather than appending onto, or mode-clashing with, the poisoned one. Other
            # env_ids' open episodes were never touched.
            self._size -= n
            raise
        if ep.is_torch:
            # torch.stack + one .to(): dtype/device conversion (if any is even
            # needed -- a same-device, same-dtype stack is a no-op copy into
            # the ring only) happens ONCE for the whole episode, never per row,
            # and never through .cpu()/.numpy() -- rows already on the learner
            # device are never pulled back to host to be staged or stored.
            self._put_obs(start, torch.stack(ep.obs))
        else:
            obs_arr = np.asarray(ep.obs, self.obs_dtype)       # (n + 1, obs_dim)
            self._put_obs(start, torch.from_numpy(obs_arr))
        act_arr = np.asarray(ep.actions, np.float32)       # (n, action_dim)
        rew_arr = np.asarray(ep.rewards, np.float32)       # (n,)
        mask_arr = np.asarray(ep.masks, np.float32)        # (n,)
        # One transfer per episode -- the hot (per-step) path above never touches `device`.
        sd = self.store_device
        self.actions[start:start + n] = torch.from_numpy(act_arr).to(sd)
        self.rewards[start:start + n] = torch.from_numpy(rew_arr).to(sd)
        self.masks[start:start + n] = torch.from_numpy(mask_arr).to(sd)
        if self.goal_dim > 0:
            # After _reserve_rows (which raises before any write and has already
            # dropped the goals of the episodes these rows evicted): a failed
            # end_episode never reaches this line, so it stores no goal.
            self._goals[start] = ep.goal
        self._episodes.append((start, n, ep.mode))

    def _reserve_rows(self, n: int) -> int:
        """Choose the start row for n+1 new rows, wrapping and evicting overlapping
        (whole, FIFO) episodes as needed. Raises if the episode can never fit."""
        rows_needed = n + 1
        if rows_needed > self.capacity:
            raise ValueError(
                f"episode of {n} transitions needs {rows_needed} rows > capacity {self.capacity}"
            )
        start = self._write
        if start + rows_needed > self.capacity:
            start = 0   # wrap; rows [self._write, capacity) become an unused gap
        end = start + rows_needed

        # A full scan (not just the FIFO front) is required for correctness: with
        # variable-size episodes, a wrap can leave an OLDER episode sitting untouched far
        # ahead of the cursor while a NEWER one (written earlier this lap, at a lower row)
        # is what the next write actually collides with -- insertion order is not always
        # row-position order once more than one lap's leftovers are alive at once.
        survivors = []
        for ev_start, ev_n, ev_mode in self._episodes:
            ev_end = ev_start + ev_n + 1
            if ev_start < end and start < ev_end:   # half-open range overlap
                self._size -= ev_n
                self._goals.pop(ev_start, None)     # the goal dies with its rows
            else:
                survivors.append((ev_start, ev_n, ev_mode))
        self._episodes = survivors

        self._write = end
        return start

    def sample_chunks(self, n_chunks: int, length: int, burn_in: int) -> Dict[str, torch.Tensor]:
        if not self._episodes:
            raise ValueError(
                f"DeviceEpisodeBuffer has no ended episodes to sample "
                f"(size {self._size}, sampleable {self.sampleable}); gate on .sampleable"
            )
        B, K, L = int(n_chunks), int(burn_in), int(length)
        T = L - K
        if T <= 0:
            raise ValueError(f"length ({L}) must exceed burn_in ({K})")

        by_mode = {0: [], 1: []}
        for start, n, mode in self._episodes:
            by_mode[mode].append((start, n))

        n_goal = min(B, max(0, round(B * self.goal_batch_frac)))
        n_explore = B - n_goal
        if not by_mode[1]:
            n_goal, n_explore = 0, B
        if not by_mode[0]:
            n_goal, n_explore = B, 0

        starts = np.empty(B, dtype=np.int64)
        ns = np.empty(B, dtype=np.int64)
        t0 = np.empty(B, dtype=np.int64)
        mode_out = np.empty(B, dtype=np.float32)

        pos = 0
        for count, mode_value in ((n_goal, 1), (n_explore, 0)):
            if count == 0:
                continue
            eps = by_mode[mode_value]
            ep_starts = np.array([s for s, _ in eps], dtype=np.int64)
            ep_ns = np.array([e_n for _, e_n in eps], dtype=np.int64)
            weights = ep_ns.astype(np.float64)
            weights /= weights.sum()
            chosen = self.rng.choice(len(eps), size=count, p=weights)
            chosen_ns = ep_ns[chosen]
            # Every row is a window start equally often: t0 ~ Uniform[0, n - 1].
            chosen_t0 = self.rng.integers(0, chosen_ns)
            sl = slice(pos, pos + count)
            starts[sl] = ep_starts[chosen]
            ns[sl] = chosen_ns
            t0[sl] = chosen_t0
            mode_out[sl] = mode_value
            pos += count

        # obs rows fetched: [t0 - K, t0 + T] (length K + T + 1); outside [0, n] -> zero.
        j_obs = np.arange(K + T + 1)
        r_obs = t0[:, None] - K + j_obs[None, :]                         # (B, K+T+1)
        obs_valid = (r_obs >= 0) & (r_obs <= ns[:, None])
        obs_idx = starts[:, None] + np.clip(r_obs, 0, ns[:, None])

        # transitions t0 .. t0+T-1; valid = 0 past n - 1 (the last row has no transition).
        i_t = np.arange(T)
        r_t = t0[:, None] + i_t[None, :]                                 # (B, T)
        t_valid = r_t <= (ns[:, None] - 1)
        t_idx = starts[:, None] + np.clip(r_t, 0, ns[:, None] - 1)

        # keep[:, j] for j = 0..K: column j refers to episode row r = t0 - K + j; 0 iff
        # r <= 0, else 1. Columns 0..K-1 gate the K burn-in steps (h *= keep[:,j] before
        # step j reads obs[:,j]); the extra column K (row r = t0) gates entry into the
        # trained window, so a window starting at episode row 0 (t0 == 0) also zeroes the
        # hidden state right before that row -- matching what act() sees at reset -- instead
        # of carrying forward whatever the last (padding) burn-in step produced.
        j_keep = np.arange(K + 1)
        r_keep = t0[:, None] - K + j_keep[None, :]                       # (B, K+1)
        keep = (r_keep > 0).astype(np.float32)

        dev, sd = self.device, self.store_device
        # Gather on the STORE device (host RAM when store_device="cpu"), then
        # ship only the sampled rows to the learner: B*(K+T+1) rows, ~19 MB
        # at 8x49 with uint8 frames, against a ring of tens of GB.
        obs_idx_t = torch.from_numpy(obs_idx).to(device=sd, dtype=torch.long)
        obs_valid_t = torch.from_numpy(obs_valid.astype(np.float32)).to(dev).unsqueeze(-1)
        rest = self.obs[obs_idx_t].to(device=dev, dtype=torch.float32)
        if self.u8_cols:
            pix = self.obs_u8[obs_idx_t].to(device=dev, dtype=torch.float32) / 255.0
            obs = torch.cat([pix, rest], dim=-1) * obs_valid_t
        else:
            obs = rest * obs_valid_t

        t_idx_t = torch.from_numpy(t_idx).to(device=sd, dtype=torch.long)
        valid_t = torch.from_numpy(t_valid.astype(np.float32)).to(dev)
        actions = self.actions[t_idx_t].to(dev) * valid_t.unsqueeze(-1)
        rewards = self.rewards[t_idx_t].to(dev) * valid_t
        masks = self.masks[t_idx_t].to(dev) * valid_t

        out = {
            "obs": obs,
            "keep": torch.from_numpy(keep).to(dev),
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "valid": valid_t,
            "mode": torch.from_numpy(mode_out).to(dev),
        }
        if self.goal_dim > 0:
            # One row per episode, looked up by the chunk's episode start row.
            out["goal"] = torch.stack([self._goals[int(s)] for s in starts]).to(
                device=dev, dtype=torch.float32)  # (B, goal_dim)
        return out
