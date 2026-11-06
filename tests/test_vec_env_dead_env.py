"""A rebuilt env's dead predecessor is KEPT, not dropped (2026-11-05).

Fine-tune job 463476: the Town01 server stalled, ThreadedVecEnv rebuilt the env
against the restarted server and training went on -- then a minute later the
process died with "terminate called after throwing an instance of
'carla::client::TimeoutException'" (rc 134), all four towns with it. The old
env, dropped at the rebuild, was collected later; a C++ destructor's call to the
replaced server timed out, and an exception thrown inside a destructor is
std::terminate, which Python cannot catch. Kept referenced, its CARLA objects
are never destroyed mid-run.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402


class _Env:
    built = []

    def __init__(self, fail_resets=0):
        self.fail_resets = fail_resets
        self.closed = False
        self.observation_space = self.action_space = None
        _Env.built.append(self)

    def reset(self, **kw):
        if self.fail_resets:
            self.fail_resets -= 1
            raise RuntimeError("time-out of 50000ms while waiting for the simulator")
        return np.zeros(2, np.float32), {}

    def close(self):
        self.closed = True


def test_a_rebuilt_envs_predecessor_stays_referenced():
    _Env.built = []
    # reset() retries once before rebuilding, so the first env fails twice
    fails = iter([2, 0])
    vec = ThreadedVecEnv([lambda: _Env(next(fails))], names=["Town01"])
    old = vec.envs[0]
    vec.reset()                              # reset fails, the retry fails -> rebuild
    assert vec.envs[0] is not old            # a fresh env serves from now on
    assert old.closed                        # the dead one was closed (best effort) ...
    assert any(e is old for e in vec._dead_envs)   # ... and kept, never garbage-collected
