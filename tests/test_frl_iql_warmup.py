"""IQL on learner C's OWN networks before the first online step (torch_rl.frl.iql.iql_warmup; user
2026-11-14: "also pretrain critic with IQL first", "ensure you borrow IQL from fastrlap's and use the
new proven learner").

The warm-up is NOT a second IQL implementation. It is IQLLearner -- the port whose five FastRLAP
rules tests/test_frl_iql.py pins and which every offline critic measurement was made with -- pointed
at the SAC learner's own modules (IQLLearner.on_learner). What can go wrong is therefore the WIRING,
and every wiring mistake is silent: the run would train either way and just be a different run.
These tests pin it:
  * the three steps are IQLLearner's own, in its order, on the learner's own critic, target critic
    and actor, and one update equals the rules written out by hand;
  * V's target and the AWR advantage read the min over ALL target heads -- not REDQ's two-head
    subset that learner C's SAC target draws -- and the learner's head generator is never used;
  * the value net sits behind its OWN copy of the critic's observation trunk, and is discarded;
  * the discount and the Polyak rate are the learner's;
  * the SAC stage starts as if IQL-pretrained weights had been loaded: the learner's three
    optimizers, its temperature and its entropy target are untouched (recorded rows never tune the
    temperature, user rule 2026-11-07).
"""
import copy
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")       # the row layout torch_rl.frl.carla is imported with
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl.frl import iql as fiql  # noqa: E402
from torch_rl.frl.learner import SACLearner  # noqa: E402

EMB, ST, ACT = 24, 2, 2


def _learner(num_qs=4, seed=0, hidden=(32, 32), **kw):
    """Learner C's layout in small: embeddings and states through the multiplexer's compression,
    LayerNorm Q heads, REDQ's target subset of two."""
    kw.setdefault("critic_layer_norm", True)
    kw.setdefault("num_min_qs", 2)
    return SACLearner(action_dim=ACT, embedding_dim=EMB, state_dim=ST, latent_dim=8, num_qs=num_qs,
                      hidden_dims=hidden, seed=seed, device="cpu", **kw)


def _batch(b=16, seed=0):
    """A batch in the replay's format (FeatureReplay.sample / ExpertReplay.sample)."""
    g = torch.Generator().manual_seed(seed)

    def obs():
        return {"embeddings": torch.randn(b, EMB, generator=g),
                "states": torch.randn(b, ST, generator=g)}
    return {"observations": obs(), "next_observations": obs(),
            "actions": torch.rand(b, ACT, generator=g) * 1.6 - 0.8,
            "rewards": torch.randn(b, generator=g),
            "masks": (torch.rand(b, generator=g) > 0.2).float()}


def _snap(module):
    return [p.detach().clone() for p in module.parameters()]


def _moved(module, before):
    return any(not torch.equal(p, q) for p, q in zip(module.parameters(), before))


def _same(a, b):
    """Deep equality of optimizer state dicts, tensors compared exactly."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    return a == b


def test_the_warmup_trains_the_learners_own_networks_and_leaves_the_sac_state_alone():
    lr = _learner()
    lr.update(_batch(seed=5), utd_ratio=1)       # one SAC step first, so the optimizers HOLD state
    crit, targ, act = lr.critic, lr.target_critic, lr.actor
    c0, t0, a0 = _snap(crit), _snap(targ), _snap(act)
    temp0, ent0 = lr.log_temp.detach().clone(), lr.target_entropy
    opts0 = {n: copy.deepcopy(getattr(lr, n).state_dict())
             for n in ("actor_opt", "critic_opt", "temp_opt")}
    gen0 = lr.gen.get_state().clone()
    info = fiql.iql_warmup(lr, lambda: _batch(), 3)
    assert lr.critic is crit and lr.target_critic is targ and lr.actor is act    # the same objects
    assert _moved(crit, c0) and _moved(targ, t0) and _moved(act, a0)
    assert torch.equal(lr.log_temp.detach(), temp0) and lr.target_entropy == ent0
    for n, st in opts0.items():
        assert _same(getattr(lr, n).state_dict(), st), f"the warm-up stepped the learner's {n}"
    assert torch.equal(lr.gen.get_state(), gen0), "REDQ's head subset must never be drawn here"
    assert info and all(math.isfinite(v) for v in info.values())


def test_a_fresh_learners_sac_optimizers_are_still_fresh_and_then_train_the_pretrained_weights():
    """The handover: SAC starts on the IQL-trained weights with Adam moments it never accumulated
    on recorded rows, exactly as if a pretrained checkpoint had been loaded into a new learner."""
    lr = _learner()
    fiql.iql_warmup(lr, lambda: _batch(), 2)
    for n in ("actor_opt", "critic_opt", "temp_opt"):
        assert getattr(lr, n).state_dict()["state"] == {}, f"{n} took a step during the warm-up"
    for m in (lr.critic, lr.actor):
        assert all(p.grad is None for p in m.parameters()), "a gradient was left behind"
    assert all(p.requires_grad for p in lr.critic.parameters())
    assert not any(p.requires_grad for p in lr.target_critic.parameters())
    c0, a0 = _snap(lr.critic), _snap(lr.actor)
    out = lr.update(_batch(seed=3), utd_ratio=1)
    assert _moved(lr.critic, c0) and _moved(lr.actor, a0) and math.isfinite(out["critic_loss"])


def test_the_warmup_runs_iqllearners_own_three_steps_in_its_order(monkeypatch):
    """Not a second implementation: every update goes through IQLLearner.update_v, update_actor and
    update_q, the methods tests/test_frl_iql.py pins and the offline measurements were made with."""
    seen = []
    for name in ("update_v", "update_actor", "update_q"):
        real = getattr(fiql.IQLLearner, name)
        monkeypatch.setattr(fiql.IQLLearner, name,
                            (lambda n, f: lambda self, b: (seen.append(n), f(self, b))[1])(name, real))
    fiql.iql_warmup(_learner(), lambda: _batch(), 2)
    assert seen == ["update_v", "update_actor", "update_q"] * 2


def test_one_warmup_update_equals_the_rules_written_out_by_hand():
    """Discount 0.9 and tau 0.3, so a hard-coded 0.99 or 0.005 cannot pass; four heads with REDQ's
    subset of two, so a subsampled min cannot pass; a temperature at which the clip binds on some
    rows only; three learning rates that are neither each other's nor the SAC optimizers'."""
    lr = _learner(num_qs=4, discount=0.9, tau=0.3)
    b = _batch(b=32)
    kw = dict(expectile=0.7, temperature=40.0, awr_clip=100.0, actor_lr=1e-3, critic_lr=6e-4,
              value_lr=5e-4)
    torch.manual_seed(7)
    val = copy.deepcopy(fiql.IQLLearner.on_learner(lr, **kw).value)     # the value net's start
    crit, targ, act = (copy.deepcopy(m) for m in (lr.critic, lr.target_critic, lr.actor))
    obs, nobs = b["observations"], b["next_observations"]
    a, r, m = b["actions"], b["rewards"], b["masks"]

    # 1. V: expectile regression on the min over ALL the target heads at the dataset action
    opt = torch.optim.Adam(val.parameters(), lr=5e-4)
    with torch.no_grad():
        q = targ(obs, a).min(dim=0).values
    diff = q.unsqueeze(0) - val(obs)
    v_loss = (torch.where(diff > 0, 0.7, 1.0 - 0.7) * diff ** 2).mean()
    opt.zero_grad()
    v_loss.backward()
    opt.step()
    # 2. actor: weighted log-likelihood of the DATASET action; weights from the target critic and
    #    the NEW V, clipped
    opt = torch.optim.Adam(act.parameters(), lr=1e-3)
    with torch.no_grad():
        adv = targ(obs, a).min(dim=0).values - val(obs).mean(dim=0)
        w = torch.clamp(torch.exp(adv * 40.0), max=100.0)
    assert float(w.max()) == 100.0 and float(w.min()) < 100.0, "the clip must bind on some rows only"
    mean, log_std = act.pi.dist_params(act.trunk(obs))
    u = torch.atanh(a.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    logp = (-0.5 * ((u - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
            - torch.log(1.0 - torch.tanh(u) ** 2)).sum(-1)
    a_loss = -(w * logp).mean()
    opt.zero_grad()
    a_loss.backward()
    opt.step()
    # 3. Q: r + discount * mask * V_target(s'), V_target a HARD copy of the new V; the squared error
    #    SUMMED over heads; then Polyak at the learner's tau
    opt = torch.optim.Adam(crit.parameters(), lr=6e-4)
    with torch.no_grad():
        y = r + 0.9 * m * val(nobs).min(dim=0).values
    q_loss = ((crit(obs, a) - y.unsqueeze(0)) ** 2).mean(dim=-1).sum()
    opt.zero_grad()
    q_loss.backward()
    opt.step()
    with torch.no_grad():
        for tp, p in zip(targ.parameters(), crit.parameters()):
            tp.mul_(1.0 - 0.3).add_(p, alpha=0.3)

    torch.manual_seed(7)                          # the warm-up builds the same value net
    info = fiql.iql_warmup(lr, lambda: b, 1, **kw)
    for name, got, want in (("critic", lr.critic, crit), ("target critic", lr.target_critic, targ),
                            ("actor", lr.actor, act)):
        for p, p_ref in zip(got.parameters(), want.parameters()):
            assert torch.allclose(p, p_ref, rtol=1e-5, atol=1e-6), name
    assert info["value_loss"] == pytest.approx(float(v_loss), rel=1e-5)
    assert info["actor_loss"] == pytest.approx(float(a_loss), rel=1e-4)
    assert info["critic_loss"] == pytest.approx(float(q_loss), rel=1e-5)
    assert info["critic_loss"] == pytest.approx(info["critic_loss_per_head"] * 4, rel=1e-5)


def test_the_value_target_is_the_min_over_ALL_target_heads_not_redqs_subset():
    lr = _learner(num_qs=4)                       # num_min_qs=2: SAC's own target draws two heads
    b = _batch()
    with torch.no_grad():
        lr.target_critic.q.heads[3][1].bias.fill_(-1000.0)     # ONE head far below the others
        want = lr.target_critic(b["observations"], b["actions"]).min(dim=0).values
    assert float(want.mean()) < -900.0
    info = fiql.iql_warmup(lr, lambda: b, 1)
    assert info["q_mean"] == pytest.approx(float(want.mean()), rel=1e-5)


def test_the_target_critic_is_polyaked_at_the_learners_own_rate():
    lr = _learner(tau=0.3)
    t0 = _snap(lr.target_critic)
    fiql.iql_warmup(lr, lambda: _batch(), 1)
    for tp, old, p in zip(lr.target_critic.parameters(), t0, lr.critic.parameters()):
        assert torch.allclose(tp, 0.7 * old + 0.3 * p.detach(), rtol=1e-5, atol=1e-7)


def test_the_value_net_has_its_own_copy_of_the_critics_trunk_and_the_critics_layer_norm():
    lr = _learner(critic_layer_norm=True)
    view = fiql.IQLLearner.on_learner(lr)
    assert view.critic is lr.critic and view.target_critic is lr.target_critic
    assert view.discount == lr.discount and view.tau == lr.tau
    shared = {id(p) for p in view.value.parameters()} & {id(p) for p in lr.critic.parameters()}
    assert not shared, "V must not share the critic's observation trunk"
    for p, q in zip(view.value.trunk.parameters(), lr.critic.trunk.parameters()):
        assert torch.equal(p, q)                  # ... but it starts as a copy of it

    def layer_norms(module):
        return sum(isinstance(x, torch.nn.LayerNorm) for x in module.modules())
    assert layer_norms(view.value.v) == 2         # one per hidden layer, as the Q heads have
    plain = fiql.IQLLearner.on_learner(_learner(critic_layer_norm=False))
    assert layer_norms(plain.value.v) == 0
    c0 = _snap(lr.critic)
    view.update_v(_batch())
    assert not _moved(lr.critic, c0), "the V step must not move the critic or its trunk"
    for p, tp in zip(view.value.parameters(), view.target_value.parameters()):
        assert torch.equal(p, tp), "target V is a hard copy, never a Polyak average"


def test_the_awr_weight_reads_the_TARGET_critic_at_the_dataset_action_and_is_clipped():
    lr = _learner()
    view = fiql.IQLLearner.on_learner(lr)
    b = _batch()
    with torch.no_grad():
        q = lr.target_critic(b["observations"], b["actions"]).min(dim=0).values
        adv = q - view.value(b["observations"]).mean(dim=0)
        w = torch.clamp(torch.exp(adv * 0.1), max=100.0)
        for p in lr.critic.parameters():          # the ONLINE critic made wildly different:
            p.add_(5.0)                           # the weights must not notice
    info = view.update_actor(b)
    assert info["adv_mean"] == pytest.approx(float(adv.mean()), rel=1e-5, abs=1e-6)
    assert info["awr_weight_mean"] == pytest.approx(float(w.mean()), rel=1e-5)
    # the clip binds only on a large positive advantage, so it is forced (as tests/test_frl_iql.py)
    hot = fiql.IQLLearner.on_learner(_learner(), temperature=1.0)
    hot.value.forward = lambda obs: torch.full((1, obs["states"].shape[0]), -1000.0)
    info = hot.update_actor(b)
    assert info["awr_weight_max"] == pytest.approx(100.0)
    assert info["awr_weight_mean"] == pytest.approx(100.0)


def test_on_a_one_step_problem_q_ranks_the_better_action_and_the_actor_follows_the_advantage():
    """masks 0 and reward -|a - a*|^2 on actions spread over the whole square: Q must come out
    higher at a* than at a far corner (true gap 2.02), and the AWR actor must sit near a* -- not at
    the dataset's mean action (0, 0), 0.58 away, where plain behaviour cloning would put it."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    n = 2048
    star = torch.tensor([0.5, -0.3])
    emb, st = torch.randn(n, EMB, generator=g), torch.randn(n, ST, generator=g)
    act = torch.rand(n, ACT, generator=g) * 1.8 - 0.9
    rew = -((act - star) ** 2).sum(-1)

    def sample(b=128):
        i = torch.randint(0, n, (b,), generator=g)
        o = {"embeddings": emb[i], "states": st[i]}
        return {"observations": o, "next_observations": o, "actions": act[i], "rewards": rew[i],
                "masks": torch.zeros(b)}

    lr = _learner(num_qs=2, num_min_qs=None, hidden=(64, 64))
    obs = {"embeddings": emb[:256], "states": st[:256]}
    with torch.no_grad():
        before = lr.actor.sample(obs)[2].mean(dim=0)
    fiql.iql_warmup(lr, sample, 400, temperature=5.0, actor_lr=3e-3, critic_lr=3e-3, value_lr=3e-3)
    with torch.no_grad():
        good = lr.critic(obs, star.expand(256, 2)).min(dim=0).values
        bad = lr.critic(obs, torch.tensor([-0.6, 0.6]).expand(256, 2)).min(dim=0).values
        after = lr.actor.sample(obs)[2].mean(dim=0)
    assert float((good > bad).float().mean()) > 0.95 and float((good - bad).mean()) > 1.0
    assert float((after - star).norm()) < 0.25 < float((before - star).norm())


def test_it_runs_on_learner_c_as_the_carla_loop_builds_it_from_the_expert_replays_batches():
    """The call the training loop makes -- iql_warmup(learner, lambda: expert.sample(batch), n,
    log=log) -- on make_learner's learner C: ten LayerNorm heads on the 50-d compressions of the
    embeddings and of [mode, speed], with the expert replay's float16 embeddings."""
    from torch_rl.frl import carla as fc
    from torch_rl.frl import expert as fx
    emb, n = 40, 64
    rng = np.random.default_rng(0)
    arrays = {"emb": rng.standard_normal((n, emb)).astype(np.float16),
              "next_emb": rng.standard_normal((n, emb)).astype(np.float16),
              "states": rng.random((n, 2)).astype(np.float32),
              "next_states": rng.random((n, 2)).astype(np.float32),
              "actions": rng.uniform(-1, 1, (n, 2)).astype(np.float32),
              "rewards": rng.standard_normal(n).astype(np.float32),
              "masks": (rng.random(n) > 0.1).astype(np.float32)}
    arrays["actions"][:8] = [1.0, -1.0]           # expert actions sit ON the bound: atanh must hold
    expert = fx.ExpertReplay(arrays, k=8, c0_md5="x", meta={}, seed=0)
    learner = fc.make_learner(emb, 2, 2, "cpu", seed=0)
    assert learner.num_qs == 10 and fc.REDQ["critic_layer_norm"]
    c0, a0 = _snap(learner.critic), _snap(learner.actor)
    lines = []
    info = fiql.iql_warmup(learner, lambda: expert.sample(32), 2, log=lines.append)
    assert _moved(learner.critic, c0) and _moved(learner.actor, a0)
    assert info and all(math.isfinite(v) for v in info.values())
    assert lines and lines[-1].startswith("[IQL] u=2/2 ")


def test_progress_is_logged_every_log_every_updates_and_at_the_end():
    lines = []
    fiql.iql_warmup(_learner(), lambda: _batch(), 5, log=lines.append, log_every=2)
    assert [line.split()[1] for line in lines] == ["u=2/5", "u=4/5", "u=5/5"]
    assert all(line.startswith("[IQL] ") and "critic_loss=" in line and "value_loss=" in line
               for line in lines)


def test_zero_updates_is_a_no_op():
    lr = _learner()
    c0, t0, a0 = _snap(lr.critic), _snap(lr.target_critic), _snap(lr.actor)
    rng0 = torch.get_rng_state().clone()

    def sample():
        raise AssertionError("no batch may be drawn for zero updates")
    assert fiql.iql_warmup(lr, sample, 0) == {}
    assert not (_moved(lr.critic, c0) or _moved(lr.target_critic, t0) or _moved(lr.actor, a0))
    assert torch.equal(torch.get_rng_state(), rng0), "not even a value net may be initialised"


def test_an_actor_that_is_not_learner_cs_own_is_refused():
    """AWR needs log pi(a | s) of the dataset action from learner C's squashed Gaussian behind its
    trunk; c0's head (FRL_ACTOR=c0, carla.C0HeadActor) has neither, and silently skipping the actor
    step would leave a run labelled IQL-pretrained with an untrained actor."""
    lr = _learner()
    lr.actor = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="actor"):
        fiql.iql_warmup(lr, lambda: _batch(), 1)
