"""Standard SAC's twin critics (user 2026-11-09: "reduce 10 critic to 1", then "use same
number standard critics use").

SAC (Haarnoja et al. 2018) keeps TWO Q-functions and uses their minimum both in the
target and in the policy loss. This arm was REDQ: N critics, the target a min over a
random subset, the actor reading their MEAN. N_CRITICS=2 with SUBSET_SIZE=2 makes the
target the min of both; ACTOR_Q_REDUCE=min makes the actor read min(Q1, Q2).
ACTOR_Q_REDUCE=mean (the default) is every earlier run.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402
from torch_rl.gated_recurrent_tqc import reduce_critics  # noqa: E402

SRC_TQC = open(os.path.join(ROOT, "torch_rl", "gated_recurrent_tqc.py")).read()
SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_mean_and_min_over_the_critic_axis():
    q = torch.tensor([[[1.0], [5.0], [-2.0]],
                      [[3.0], [4.0], [-6.0]]])          # (n_critics 2, batch 3, 1)
    assert torch.equal(reduce_critics(q, "mean"), torch.tensor([2.0, 4.5, -4.0]))
    assert torch.equal(reduce_critics(q, "min"), torch.tensor([1.0, 4.0, -6.0]))
    with pytest.raises(ValueError, match="ACTOR_Q_REDUCE"):
        reduce_critics(q, "max")


def test_the_default_is_the_old_mean():
    assert GatedRecurrentSAC(_cfg()).cfg.actor_q_reduce == "mean"


def test_the_actor_loss_reads_the_reduction():
    upd = SRC_TQC[SRC_TQC.index("def _update_impl"):]
    upd = upd[:upd.index("\n    def ", 10)]
    assert "reduce_critics(self._q(self.critics_g, s_d, st, self.goal_cols, new_a)" in upd
    assert ".mean(dim=(0, 2))\n" not in upd.split("if has_g:")[2]   # the actor's own Q


def test_two_critics_with_the_min_train():
    a = GatedRecurrentSAC(_cfg(n_critics=2, subset_size=2, actor_q_reduce="min",
                               critic_layer_norm=True))
    w = a.actor.trunk[0].weight.detach().clone()
    for s in range(3):
        m = a.update(_chunk(a.cfg, [1, 0, 1, 0], seed=s))
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in m.values()
               if isinstance(v, (int, float)))
    assert not torch.equal(w, a.actor.trunk[0].weight)
    assert tuple(a.critics_g.weights[0].shape)[0] == 2


def test_the_trainer_reads_the_setting():
    assert 'actor_q_reduce=os.environ.get("ACTOR_Q_REDUCE", "mean")' in SRC
