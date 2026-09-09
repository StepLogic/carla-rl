"""Threaded vectorised CARLA envs, one town per env.

WHY NOT gymnasium's SyncVectorEnv OR AsyncVectorEnv.

  * SyncVectorEnv steps envs one after another. Each CARLA step is a blocking
    RPC to a server process: the client spends nearly all of it waiting on a
    socket, not computing. Stepping four envs in sequence therefore costs four
    round-trips of latency for four transitions -- no throughput gain at all,
    only decorrelation.

  * AsyncVectorEnv forks a subprocess per env. A CARLA client holds a live RPC
    connection and C++ state that does not survive a fork cleanly, and each
    subprocess would need its own connection anyway.

  * A THREAD pool is the right shape here. The GIL is released during socket
    I/O, so N env steps overlap their waits and the vec step costs roughly ONE
    round-trip instead of N. This is the rare case where threads genuinely
    parallelise Python work, because almost none of the work is Python.

PINNING A TOWN PER ENV avoids reloading the world every episode, which is what
the original single-env trainer did -- tens of seconds of map load for a few
hundred steps of data. With enough servers every batch is also mixed across
towns, which a held-out-town claim wants more than raw step count.

WHEN THE CARD CANNOT HOLD ONE SERVER PER TOWN, replace() rotates an env onto a
different town at an episode boundary. That restores town COVERAGE at the cost
of town MIXING: within a rotation block the buffer fills from a single map. It
is a compromise forced by VRAM (measured ~4.9-5.5 GB per CARLA server, nearly
independent of quality or headless), not a preference.

EACH ENV OWNS ITS EPISODE. Environments finish at different times, so episode
boundaries are per-env and surfaced in the step result. The caller uses them to
flush that env's HER buffer -- a shared episode boundary would splice
transitions from different towns into one hindsight trajectory.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ThreadedVecEnv:
    """Steps N CARLA envs concurrently. Auto-resets each on termination."""

    def __init__(self, env_fns, names=None):
        self.n = len(env_fns)
        self.names = list(names) if names else [f"env{i}" for i in range(self.n)]
        # Built in parallel too: each CarlaEnv construction loads a map, and
        # doing that serially for four towns is minutes of dead time.
        self._pool = ThreadPoolExecutor(max_workers=self.n)
        # Factories are RETAINED so a broken env can be rebuilt. A CARLA sensor
        # fault can destroy the hero actor, after which reset() raises
        #   Actor could not be found in the registry
        # forever -- the env is unrecoverable and only a rebuild restores it.
        self._env_fns = list(env_fns)
        self.envs = list(self._pool.map(lambda f: f(), env_fns))

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._last_obs = [None] * self.n
        self.episode_returns = np.zeros(self.n, np.float64)
        self.episode_lengths = np.zeros(self.n, np.int64)

    def _safe_reset(self, i):
        """reset() env i, rebuilding it if the actors are gone.

        Two failure levels: a transient reset error (retry), and a destroyed
        actor registry (rebuild). Both were seen in one run -- the step fault
        handler recovered, then the auto-reset raised uncaught and killed a
        run that had already survived the original fault.
        """
        try:
            out = self.envs[i].reset()
            return out if isinstance(out, tuple) else (out, {})
        except Exception as e:  # noqa: BLE001
            print(f"[VecEnv] env{i} ({self.names[i]}) reset failed: "
                  f"{type(e).__name__}: {e} -- rebuilding env", flush=True)
        try:
            self.envs[i].close()
        except Exception:  # noqa: BLE001
            pass
        self.envs[i] = self._env_fns[i]()
        out = self.envs[i].reset()
        return out if isinstance(out, tuple) else (out, {})

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed=None):
        def _reset(i):
            try:
                kw = {} if seed is None else {"seed": int(seed) + i}
                out = self.envs[i].reset(**kw)
                return out if isinstance(out, tuple) else (out, {})
            except Exception:  # noqa: BLE001
                return self._safe_reset(i)

        results = list(self._pool.map(_reset, range(self.n)))
        self._last_obs = [o for o, _ in results]
        infos = [inf for _, inf in results]
        self.episode_returns[:] = 0.0
        self.episode_lengths[:] = 0
        return self._stack(self._last_obs), infos

    def step(self, actions):
        """actions: (n, action_dim). Returns stacked obs and per-env lists.

        `dones` marks episode END for that env. The observation returned for a
        finished env is the RESET observation (auto-reset), while `final_infos`
        carries the terminal info -- so a caller can both record the terminal
        transition and continue stepping without a bookkeeping stall.
        """
        def _step(i):
            # SURVIVE A TRANSIENT SIM FAULT. CARLA raises
            #   RuntimeError: A sensor took too long to send their data
            # when a sensor misses its deadline; it surfaced as queue.Empty
            # inside this pool and killed a 200k-step run at step 25k, right
            # after a 60-episode evaluation. A dropped frame is not a reason to
            # lose a training run -- the env is reset and the episode
            # abandoned, exactly as the collection stage retries a crashed
            # server.
            #
            # The transition is marked env_error so the caller DISCARDS it
            # rather than learning from a synthetic one.
            try:
                return self.envs[i].step(np.asarray(actions[i]))
            except Exception as e:  # noqa: BLE001
                print(f"[VecEnv] env{i} ({self.names[i]}) step failed: "
                      f"{type(e).__name__}: {e} -- resetting, episode abandoned",
                      flush=True)
                try:
                    obs_i, _ = self._safe_reset(i)
                except Exception as e2:  # noqa: BLE001
                    print(f"[VecEnv] env{i} rebuild also failed: {e2}", flush=True)
                    obs_i = self._last_obs[i]
                return obs_i, 0.0, False, True, {"env_error": True}

        results = list(self._pool.map(_step, range(self.n)))

        obs, rewards, dones, infos, final_infos = [], [], [], [], []
        # Info from the AUTO-RESET, not from the terminal step. A caller that
        # re-plans per episode (the expert oracle) needs the NEW episode's
        # goal; final_infos carries the OLD one, and planning a route to a
        # stale goal is indistinguishable from the oracle not running at all.
        reset_infos = [None] * self.n
        reset_idx = []
        for i, (o, r, term, trunc, inf) in enumerate(results):
            done = bool(term or trunc)
            self.episode_returns[i] += float(r)
            self.episode_lengths[i] += 1
            rewards.append(float(r))
            # `terminated` vs `truncated` matters to the critic: a time-limit
            # truncation must still bootstrap, a real termination must not.
            inf = dict(inf)
            inf["terminated"] = bool(term)
            inf["truncated"] = bool(trunc)
            infos.append(inf)
            dones.append(done)
            if done:
                inf["episode_return"] = float(self.episode_returns[i])
                inf["episode_length"] = int(self.episode_lengths[i])
                final_infos.append(inf)
                reset_idx.append(i)
            else:
                final_infos.append(None)
            obs.append(o)

        if reset_idx:
            for i, (o, rinf) in zip(reset_idx,
                                    self._pool.map(self._safe_reset, reset_idx)):
                obs[i] = o
                reset_infos[i] = rinf
                self.episode_returns[i] = 0.0
                self.episode_lengths[i] = 0

        self._last_obs = obs
        self.reset_infos = reset_infos
        return (self._stack(obs), np.asarray(rewards, np.float32),
                np.asarray(dones, bool), infos, final_infos)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _stack(obs_list):
        """List of Dict observations -> Dict of stacked arrays."""
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return np.stack(obs_list)

    @staticmethod
    def unstack(stacked, i):
        """One env's observation out of a stacked batch."""
        if isinstance(stacked, dict):
            return {k: v[i] for k, v in stacked.items()}
        return stacked[i]

    def replace(self, i, env_fn, name=None):
        """Swap env i for a freshly built one (used for town rotation).

        Closing and rebuilding is how a single server covers several towns: the
        new env's construction triggers the CARLA world load. Returns that
        env's first observation so the caller can splice it into the stacked
        batch without a full vec reset -- the other envs are mid-episode and
        must not be disturbed.
        """
        try:
            self.envs[i].close()
        except Exception:  # noqa: BLE001
            pass
        self.envs[i] = env_fn()
        if name:
            self.names[i] = name
        out = self.envs[i].reset()
        obs, info = out if isinstance(out, tuple) else (out, {})
        self._last_obs[i] = obs
        self.episode_returns[i] = 0.0
        self.episode_lengths[i] = 0
        return obs, info

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:  # noqa: BLE001
                pass
        self._pool.shutdown(wait=False)
