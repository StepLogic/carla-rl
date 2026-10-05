"""TQC with one actor and TWO critics, selected by a commanded mode bit.

THE SHAPE OF THE IDEA. Two MDPs share dynamics and an action space but not a
reward. One actor serves both, told which it is in by a bit inside the
observation. Each mode gets its own critic, its own discount and its own
temperature; the batch carries both modes and each critic's loss is MASKED down
to its own rows. This is multi-task SAC with a task indicator, not a scalarised
sum of two rewards.

WHY TWO CRITICS HERE. The answer DEPENDS ON THE GOAL REWARD, and both cases are
live in this repo, selected by GATED_GOAL_REWARD.

  sorb (default)  The goal arm pays -1 per step until arrival and runs at
                  gamma = 1.0, so its Q IS the cost-to-go in steps, bounded in
                  [-step_cost * cap, 0]. The explore arm pays a lane-progress
                  bonus at gamma = 0.99 and is bounded only by its own discount.
                  The discounts genuinely differ, so one critic could not serve
                  both even at equal scale, and an undiscounted critic fitted to
                  the explore arm would have no bounded fixed point at all. This
                  is the Habitat peer's rationale, and under this reward it does
                  transfer.

  shaped          The goal arm pays delta_distance + projected_distance and both
                  arms run at 0.99. The discounts are then EQUAL and the
                  Habitat rationale does NOT apply; the split is justified by
                  RETURN SCALE alone, order 100 against order 10, and by the
                  Huber knee `kappa` being derived from that scale. A single
                  critic fitted to both is dominated by the larger.

An earlier version of this file asserted the second case unconditionally. That
was correct for the shaped reward and wrong the moment the reward changed, which
is why the reasoning is spelled out per case rather than stated once.

gamma = 1.0 IS ONLY SAFE BECAUSE EVERY EPISODE ENDS. Under sorb the goal return
is bounded in [-step_cost * max_steps, 0] by construction, and the goal critic
is clipped to that window. The clip is PER CRITIC, and the explore critic must
never see it: that window is derived from the GOAL arm's reward and episode cap
and says nothing about the explore arm's range.

MASKED LOSSES, NEVER A PER-SAMPLE DISCOUNT. Each critic computes its target for
every row at its OWN scalar discount and then zeroes the off-mode rows out of
its loss, so those rows contribute exactly zero gradient. The tempting
alternative -- one critic with a per-row gamma vector -- destroys the goal
critic's value semantics: its Q can no longer be read as a distance, and the map
builder reads it as exactly that.

WHY THIS SUBCLASSES FROM OUTSIDE THE VENDORED FORK. `dependencies/jaxrl2` is a
nested repository placed on sys.path by ABSOLUTE path in easy-install.pth, so it
is shared by every session on this machine and cannot be isolated in a git
worktree. Subclassing from carla-rl keeps the fork unpatched, keeps this code
under this repo's history, and means a half-written edit can never reach another
session's training run.

WHAT IS DELIBERATELY IDENTICAL to the two specialist arms: the encoder,
augmentation, PixelMultiplexer paths, target smoothing, truncation and kappa all
come from TQCLearner untouched. The ablation compares gated against goal-only
against goal-free, so the gated arm must differ by the GATE and nothing else.
"""
import copy
import functools
from typing import Dict, Optional

# FIRST, and for the side effect: aliases jax.core.Var for distrax 0.1.5 against
# jax 0.6.2. The rescaled-tanh policy path needs it, and that is the path this
# learner takes. Must precede the policy's construction.
import vae_nav.jax_compat  # noqa: F401

import jax
import jax.numpy as jnp
from jaxrl2.agents.sac.temperature_updater import update_temperature
from jaxrl2.agents.tqc.quantiles import tqc_target
from jaxrl2.agents.tqc.tqc_learner import TQCLearner, _share_encoder
from jaxrl2.utils.augmentations import batched_random_crop
from jaxrl2.utils.misc import augment_batch
from jaxrl2.utils.target_update import soft_target_update

from vae_nav.lane_priv import (COST_KEY, EXPERT_KEY, PRIV_KEY, strip_expert,
                               strip_priv)
from vae_nav.quantile_per_sample import (masked_mean,
                                         quantile_huber_loss_per_sample)

# Must match vae_nav.gated_reward and vae_nav.vae_goal_wrapper.
GOAL, EXPLORE = 1, 0
MODE_KEY = "mode"


def apply_lane_cost(batch, lane_cost, mu):
    """-> `batch` with rewards - mu * lane_cost, or `batch` itself if there is no cost.

    Pure, and never mutates its input: the explore critic is updated from the same
    batch afterwards and must see the unpenalised reward. Called inside the jitted
    update with mu as a TRACED argument, so a new multiplier every update costs no
    recompile. See lane_priv.COST_KEY for why the cost is priced here and not when
    the row is stored.
    """
    if lane_cost is None:
        return batch
    return {**batch, "rewards": batch["rewards"] - mu * lane_cost}


def _mode_mask(batch):
    """Goal-row indicator, read from the OBSERVATION rather than the batch.

    The replay buffer strips its routing key on insert, so the mode does not
    survive as a batch field. It does not need to: the bit is an actor input,
    so it is already in the observation at a known key, and reading it there
    means the learner needs no change to the batch schema.
    """
    return (batch["observations"][MODE_KEY][:, 0] > 0.5).astype(jnp.float32)


def _masked_critic_update(key, actor, critic, target_critic_params, temp, batch,
                          discount, backup_entropy, top_quantiles_to_drop,
                          kappa, mask, n_steps=None,
                          value_floor=None, value_ceiling=None,
                          alpha_scale=1.0):
    """One critic's update, restricted to the rows `mask` selects.

    value_floor / value_ceiling clip this critic's TARGET only, and are applied
    PER CRITIC. Under the SORB minimum-time reward the goal critic runs at
    gamma = 1.0, where V(s) = -c + V(s') has no bounded fixed point on a state
    the agent never escapes; the bound makes the return range explicit
    ([-step_cost * max_steps, 0]) so a target outside it is unrealisable rather
    than merely unlikely.

    THEY CANNOT COME FROM THE GLOBAL TQC_TARGET_MIN/MAX. Those are read at
    import in critic_updater and baked into every trace, so they would impose
    the GOAL arm's window on the EXPLORE critic too. That window is derived from
    the goal reward and the goal episode cap and has no bearing on the explore
    arm's range, so it would be an arbitrary constraint on a critic it was not
    computed for -- loose today, and able to bind silently in either direction
    once the explore policy starts earning progress. The explore critic passes
    None here and is left unclipped.
    """
    target_critic = critic.replace(params=target_critic_params)
    # THE ACTOR NEVER SEES THE PRIVILEGED BLOCK (a no-op without OBS_PRIV).
    # The critic below reads whatever batch it was handed: the privileged
    # goal critic gets the full observation, every other critic a stripped one.
    dist = actor.apply_fn({"params": actor.params},
                          strip_priv(batch["next_observations"]))
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_quantiles = target_critic.apply_fn(
        {"params": target_critic_params}, batch["next_observations"], next_actions)
    # alpha_scale (QNORM_CRITIC_ALPHA, default 1.0 = off). Under ACTOR_Q_NORM
    # the actor divides its -Q term by the mode's running value spread s, so
    # the temperature it effectively experiences is alpha * s -- while this
    # backup still soft-maximises against plain alpha, and the two disagree
    # by exactly the factor the normalisation introduced. Scaling the
    # CRITIC'S entropy term by the same s makes the backup match. Default 1.0
    # multiplies by one and the target is byte-identical to the pre-flag
    # computation.
    alpha = (alpha_scale * temp.apply_fn({"params": temp.params})
             if backup_entropy else 0.0)

    # THE TARGET IS COMPUTED FOR EVERY ROW, at this critic's own discount, and
    # the wrong-mode rows are discarded by the mask below. That wastes compute
    # on half the batch and is deliberate: the alternative is dynamic control
    # flow inside a jitted function, whose shapes the compiler cannot fix.
    #
    # N-STEP BOOTSTRAP DISCOUNT (v11). With n-step returns the batch carries a
    # per-row `n_steps` = the ACTUAL merged segment length n', and the discount
    # becomes gamma**n' per row -- the buffer's reward sum was accumulated in
    # this same gamma frame, so target = sum_k gamma^k r_{t+k} + gamma^n' * Q
    # is exactly the n-step soft return. jaxrl2's tqc_target computes
    # `discount * masks[:, None] * soft`: a FLAT (batch,) discount would
    # broadcast there as (1, batch) x (batch, 1) -> (batch, batch) -- a silent
    # outer product, and an instant shape error against soft (batch, kept)
    # unless kept == batch, in which case it is quietly WRONG MATH. So the
    # per-row discount is passed pre-shaped to (batch, 1), which multiplies
    # masks column-wise exactly like the scalar does. The vendored code is
    # still untouched. With n_steps absent (n_step = 1, every earlier run) the
    # scalar path is byte-identical.
    #
    # ONE DELIBERATE SIMPLIFICATION, stated because it is a deviation from
    # chaining the 1-step soft operator: the entropy correction -alpha*log pi
    # is applied at the BOOTSTRAP state only, not at every intermediate step of
    # the horizon. Chaining would need next_log_probs at each of the n'-1
    # intermediate states under the CURRENT policy, which the buffer does not
    # and cannot store. With the temperatures auto-tuned to the entropy target
    # the correction is alpha * |log pi| ~ 0.01 per step -- two orders below the
    # -1/step reward -- so the bias is negligible next to the variance n-step
    # already accepts.
    if n_steps is None:
        step_discount = discount
    else:
        step_discount = (discount ** n_steps)[:, None]
    target = tqc_target(
        next_quantiles,
        jnp.nan_to_num(batch["rewards"]),
        batch["masks"],
        discount=step_discount,
        top_quantiles_to_drop=top_quantiles_to_drop,
        alpha=alpha,
        next_log_probs=next_log_probs,
    )
    # Applied to the TARGET only, never to the prediction: the prediction must
    # stay free to be wrong so the loss can pull it back.
    if value_floor is not None or value_ceiling is not None:
        target = jnp.clip(target, value_floor, value_ceiling)

    def critic_loss_fn(params):
        qs = critic.apply_fn({"params": params}, batch["observations"],
                             batch["actions"])
        per_sample = quantile_huber_loss_per_sample(qs, target, kappa=kappa)
        loss = masked_mean(per_sample, mask)
        q_row = qs.mean(axis=(0, 2))
        td = jnp.abs(q_row - target.mean(axis=1))
        # Statistics are masked too. An unmasked mean would report the other
        # mode's values -- and with returns of order 100 against order 10, a
        # blended number looks like a healthy critic while hiding a starved one.
        return loss, {"critic_loss": loss,
                      "q": masked_mean(q_row, mask),
                      "td_errors": masked_mean(td, mask),
                      # The trainer's [Critic] line reads this exact key and
                      # prints nan when it is missing, which reads as a broken
                      # actor rather than as absent telemetry. It cost a
                      # diagnostic detour in the first live run.
                      "target_actor_entropy": masked_mean(-next_log_probs, mask)}

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    return critic.apply_gradients(grads=grads), info


def _gated_actor_update(key, actor, critic, critic_e, temp, temp_e, batch,
                        m_goal, explore_actor_w, q_scale_g=1.0, q_scale_e=1.0,
                        batch_full=None, bc_mask=None, bc_weight=0.0,
                        bc_scale=1.0):
    """One actor, choosing its critic and its temperature PER ROW.

    q_scale_g / q_scale_e (PER-MODE Q NORMALISATION, default 1.0 = off): each
    mode's -Q term is divided by a running estimate of that critic's value
    spread before the mean, so neither critic can dominate the shared actor
    by its units alone. The actor's gradient comes from dQ/da, whose size
    scales with the critic's value range: goal values span ~750 (gamma=1, cap
    750) while explore values are whatever the explore reward scale makes them
    (v4: ~10; v5 at x10 progress: ~200-600). Left unnormalised, the larger
    range owns the shared network. The temperature auto-tunes to the target
    entropy, so dividing Q rescales nothing that alpha does not absorb.
    """
    alpha_g = temp.apply_fn({"params": temp.params})
    alpha_e = temp_e.apply_fn({"params": temp_e.params})
    is_goal = m_goal > 0.5
    m_explore = 1.0 - m_goal

    # ASYMMETRIC GOAL CRITIC (OBS_PRIV): `critic` is then the PRIVILEGED goal
    # critic and reads `batch_full` (with the lane block); the actor and the
    # explore critic read `batch`, which never carries it. Without OBS_PRIV
    # batch_full is None and this is the original computation.
    obs_goal_critic = (batch if batch_full is None else batch_full)["observations"]

    def actor_loss_fn(params):
        dist = actor.apply_fn({"params": params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        # Both critics are evaluated on the whole batch and half of each result
        # is thrown away. Same trade as above: a fixed shape the JIT can handle.
        q_g = critic.apply_fn({"params": critic.params},
                              obs_goal_critic, actions).mean(axis=(0, 2))
        q_e = critic_e.apply_fn({"params": critic_e.params},
                                batch["observations"], actions).mean(axis=(0, 2))
        q = jnp.where(is_goal, q_g / q_scale_g, q_e / q_scale_e)
        alpha = jnp.where(is_goal, alpha_g, alpha_e)
        per_row = alpha * log_probs - q
        # BEHAVIOUR CLONING ON EXPERT ROWS (BC_WEIGHT, user 2026-09-28 "i
        # prefer 2"; TD3+BC-style). The deterministic action is pulled toward
        # the action the expert took, per dimension in units of the action
        # box's half-range so steer and throttle weigh alike. Q is already
        # divided by its running spread (ACTOR_Q_NORM), so BC_WEIGHT sets the
        # balance directly. Absent mask (BC off) -> the original loss.
        bc_err = None
        if bc_mask is not None:
            a_det = dist.mode()
            bc_err = (((a_det - batch["actions"]) / bc_scale) ** 2).mean(-1)
            per_row = per_row + bc_weight * bc_mask * bc_err
        # the raw per-mode value spread, for the caller's running normaliser
        def _masked_std(x, m):
            n = jnp.maximum(m.sum(), 1.0)
            mu = (x * m).sum() / n
            return jnp.sqrt(((x - mu) ** 2 * m).sum() / n)
        q_std_g = jax.lax.stop_gradient(_masked_std(q_g, m_goal))
        q_std_e = jax.lax.stop_gradient(_masked_std(q_e, m_explore))
        # explore_actor_w damps the explore-mode gradient during warm-up. After
        # a warm start the explore critic is stale, and an actor following a
        # stale critic moves away from a policy that was already good.
        loss = jnp.where(is_goal, per_row, explore_actor_w * per_row).mean()
        return loss, {
            "actor_loss": loss,
            "entropy": -log_probs.mean(),
            "entropy_goal": masked_mean(-log_probs, m_goal),
            "entropy_explore": masked_mean(-log_probs, m_explore),
            "q_std_goal": q_std_g, "q_std_explore": q_std_e,
            **({} if bc_err is None else {
                "bc_loss": masked_mean(bc_err, bc_mask),
                "bc_frac": bc_mask.mean()}),
        }

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    return actor.apply_gradients(grads=grads), info


@functools.partial(
    jax.jit,
    static_argnames=("backup_entropy", "top_quantiles_to_drop", "kappa",
                     "target_entropy", "target_entropy_e", "value_floor",
                     "value_ceiling", "enable_update_temperature", "augument",
                     "share_encoder"))
def _gated_update_jit(rng, actor, critic, target_critic_params, temp,
                      critic_e, target_critic_e_params, temp_e,
                      batch, discount, discount_e, tau, explore_actor_w,
                      target_entropy, target_entropy_e, backup_entropy,
                      top_quantiles_to_drop, kappa,
                      value_floor=None, value_ceiling=None,
                      enable_update_temperature=True, augument=True,
                      share_encoder=True, q_scale_g=1.0, q_scale_e=1.0,
                      alpha_scale_g=1.0, alpha_scale_e=1.0,
                      critic_p=None, target_critic_p_params=None,
                      bc_weight=0.0, bc_scale=1.0, lagrange_mu=0.0):
    if share_encoder:
        # From the GOAL critic, matching the ungated path exactly. Sharing from
        # both would need a rule for combining them, and any such rule is a
        # difference from the specialist arms that the ablation would then be
        # measuring alongside the gate.
        actor = _share_encoder(source=critic, target=actor)

    rng, key = jax.random.split(rng)
    if augument:
        # A no-op for these observations -- augmentation only touches arrays of
        # three or more dimensions and every key here is a flat vector -- but
        # kept so the gated path is byte-for-byte the specialists' pipeline.
        rng, batch = augment_batch(key, batch, batched_random_crop)

    # THE TWO VIEWS (OBS_PRIV). `batch_full` keeps the privileged lane block
    # for the privileged goal critic; `batch` drops it for the actor, the
    # image-only goal critic and the explore critic. Without the block both
    # are the same dict and every line below traces as before.
    bc_mask = (batch["observations"][EXPERT_KEY][:, 0].astype(jnp.float32)
               if EXPERT_KEY in batch["observations"] else None)
    # LANE COST (LAGRANGE_LANE). Read before the strips below remove it. Key
    # presence is a static property of the batch, so this resolves at trace time.
    lane_cost = (batch["observations"][COST_KEY][:, 0].astype(jnp.float32)
                 if COST_KEY in batch["observations"] else None)
    batch_full = strip_expert(batch)
    batch = strip_priv(batch)
    m_goal = _mode_mask(batch)
    m_explore = 1.0 - m_goal

    # N-STEP (v11): the batch's per-row merged segment length, or None on the
    # 1-step path. Key presence is a STATIC property of the batch structure, so
    # this branch resolves at trace time and the jit never sees dynamic control
    # flow. Both critics read the SAME array: it already encodes each row's
    # own mode's n', and the row's own critic is the only one whose loss keeps
    # it, so no per-critic split is needed.
    n_steps = batch["n_steps"] if "n_steps" in batch else None

    # THE MULTIPLIER IS CHARGED TO THE GOAL SIDE ONLY. The explore critic keeps
    # `batch` -- its reward has its own lane terms and is not under this
    # constraint. Goal rows carrying a cost are charged mu per fined crossing.
    if lane_cost is not None and n_steps is not None:
        # An n-step row's reward is a SUM over up to n transitions, but the cost
        # here is one transition's. Charging mu once would under-price every
        # merged row by however many crossings its tail held.
        raise ValueError("LAGRANGE_LANE is not defined for n-step returns: the "
                         "stored cost is per transition, the reward is not")
    batch_g = apply_lane_cost(batch, lane_cost, lagrange_mu)
    batch_full_g = apply_lane_cost(batch_full, lane_cost, lagrange_mu)

    rng, key_g = jax.random.split(rng)
    # GOAL CRITIC: bounded, because under the SORB reward it runs at gamma=1.0.
    new_critic, info_g = _masked_critic_update(
        key_g, actor, critic, target_critic_params, temp, batch_g, discount,
        backup_entropy, top_quantiles_to_drop, kappa, m_goal,
        n_steps=n_steps,
        value_floor=value_floor, value_ceiling=value_ceiling,
        alpha_scale=alpha_scale_g)

    # EXPLORE CRITIC: UNBOUNDED, deliberately.
    #
    # THE REASON IS THE WINDOW'S PROVENANCE, NOT THE SIGN OF THE RETURNS. An
    # earlier version of this comment said the explore targets are positive and
    # would therefore be clipped to zero. Measured early in training they are
    # NEGATIVE (about -63 over ~180 steps), dominated by the steer-change
    # penalty before the policy earns any progress -- so that specific argument
    # was wrong.
    #
    # The correct reason: [-step_cost * goal_max_steps, 0] is derived from the
    # GOAL arm's reward and episode cap, and says nothing about the explore
    # arm's range. The explore arm is discounted at 0.99, so its return is
    # bounded by its own geometry and needs no external clip; imposing the goal
    # arm's window on it would be an arbitrary constraint that happens to be
    # loose today and could silently bind later, in either direction.
    rng, key_e = jax.random.split(rng)
    new_critic_e, info_e = _masked_critic_update(
        key_e, actor, critic_e, target_critic_e_params, temp_e, batch,
        discount_e, backup_entropy, top_quantiles_to_drop, kappa, m_explore,
        n_steps=n_steps, alpha_scale=alpha_scale_e)

    # PRIVILEGED GOAL CRITIC (OBS_PRIV, user 2026-09-27): the SAME reward,
    # discount, clip, temperature and goal-row mask as the image-only goal
    # critic above -- it differs only in seeing the lane block. It is the one
    # the actor follows on goal rows; the image-only critic keeps training on
    # the same batch, against the same actor, and is what the map builder
    # reads as a distance (agent._critic).
    new_critic_p, new_target_p_params, info_p = None, None, None
    if critic_p is not None:
        rng, key_p = jax.random.split(rng)
        new_critic_p, info_p = _masked_critic_update(
            key_p, actor, critic_p, target_critic_p_params, temp, batch_full_g,
            discount, backup_entropy, top_quantiles_to_drop, kappa, m_goal,
            n_steps=n_steps,
            value_floor=value_floor, value_ceiling=value_ceiling,
            alpha_scale=alpha_scale_g)
        new_target_p_params = soft_target_update(
            new_critic_p.params, target_critic_p_params, tau)

    new_target_params = soft_target_update(
        new_critic.params, target_critic_params, tau)
    new_target_e_params = soft_target_update(
        new_critic_e.params, target_critic_e_params, tau)

    rng, key_a = jax.random.split(rng)
    new_actor, info_a = _gated_actor_update(
        key_a, actor, new_critic if critic_p is None else new_critic_p,
        new_critic_e, temp, temp_e, batch,
        m_goal, explore_actor_w, q_scale_g, q_scale_e,
        batch_full=None if critic_p is None else batch_full,
        bc_mask=bc_mask, bc_weight=bc_weight, bc_scale=bc_scale)

    new_temp, new_temp_e = temp, temp_e
    alpha_info = {}
    if enable_update_temperature:
        # A MODE ABSENT FROM THE BATCH TAKES A ZERO STEP. Substituting the
        # target for the measured entropy makes the temperature loss exactly
        # zero, so alpha is not dragged on evidence that does not exist. At the
        # start of training only one mode has been visited.
        has_g = m_goal.sum() > 0
        has_e = m_explore.sum() > 0
        ent_g = jnp.where(has_g, info_a["entropy_goal"], target_entropy)
        ent_e = jnp.where(has_e, info_a["entropy_explore"], target_entropy_e)
        new_temp, ag = update_temperature(temp, ent_g, target_entropy)
        new_temp_e, ae = update_temperature(temp_e, ent_e, target_entropy_e)
        alpha_info = {**ag, **{f"{k}_e": v for k, v in ae.items()}}

    info = {
        **info_a,
        **alpha_info,
        "critic_loss": info_g["critic_loss"], "q": info_g["q"],
        "td_errors": info_g["td_errors"],
        # THE KEY THE TRAINER'S [Critic] LINE PRINTS. Computing it in
        # _masked_critic_update was not enough: this dict is assembled key by
        # key, so anything not named here is silently dropped and the trainer
        # falls back to nan -- which reads as a diverged actor. That is exactly
        # what happened after the first attempt at this fix.
        "target_actor_entropy": info_g["target_actor_entropy"],
        "critic_loss_e": info_e["critic_loss"], "q_e": info_e["q"],
        "td_errors_e": info_e["td_errors"],
        "target_actor_entropy_e": info_e["target_actor_entropy"],
        "frac_goal": m_goal.mean(),
    }
    if info_p is not None:
        info.update({"critic_loss_p": info_p["critic_loss"], "q_p": info_p["q"],
                     "td_errors_p": info_p["td_errors"]})
    return (rng, new_actor, new_critic, new_target_params, new_temp,
            new_critic_e, new_target_e_params, new_temp_e, info,
            new_critic_p, new_target_p_params)


class GatedTQCLearner(TQCLearner):
    """TQCLearner with a second critic and temperature for the explore mode."""

    def __init__(self, seed, observations, actions, discount_e: float = 0.99,
                 init_temperature_e: Optional[float] = None,
                 target_entropy_e: Optional[float] = None,
                 explore_actor_warmup: int = 0,
                 actor_freeze_updates: int = 0,
                 actor_q_norm: bool = False,
                 actor_q_norm_tau: float = 0.01,
                 qnorm_critic_alpha: bool = False,
                 value_floor: Optional[float] = None,
                 value_ceiling: Optional[float] = None,
                 bc_weight: float = 0.0, **kwargs):
        # ASYMMETRIC GOAL CRITIC. With the privileged lane block present the
        # actor, the image-only goal critic and the explore critic are built
        # WITHOUT it, so none of them has an input layer that could read it.
        self.asymmetric = hasattr(observations, "keys") and PRIV_KEY in observations
        # BC weight and the action half-range its error is measured in.
        self.bc_weight = float(bc_weight)
        _asp = kwargs.get("action_space")
        self._bc_scale = (jnp.asarray(0.5 * (_asp.high - _asp.low), jnp.float32)
                          if _asp is not None else jnp.ones((), jnp.float32))
        super().__init__(seed, strip_priv(observations), actions, **kwargs)

        # GOAL CRITIC ONLY. Under the SORB minimum-time reward the goal return
        # lies in [-step_cost * max_steps, 0] BY CONSTRUCTION, so a target
        # outside that is unrealisable and clipping adds information rather than
        # bias. The explore critic is never clipped -- see _gated_update_jit.
        self.value_floor = None if value_floor is None else float(value_floor)
        self.value_ceiling = (None if value_ceiling is None
                              else float(value_ceiling))
        self.discount_e = float(discount_e)
        self.target_entropy_e = (float(self.target_entropy)
                                 if target_entropy_e is None
                                 else float(target_entropy_e))
        self.explore_actor_warmup = int(explore_actor_warmup)
        # ACTOR FREEZE (warm starts). For the first `actor_freeze_updates`
        # updates only the critics and their targets move; the actor and both
        # temperatures are held. Measured 2026-09-11 on v5, warm-started from
        # v4 with a fresh buffer and a re-scaled explore reward: the actor's
        # goal-conditioned steering fell from corr 0.57 to 0.04 within 50k
        # steps because it was updated against two critics far from their
        # fixed points, then took >75k steps to crawl back to 0.23.
        self.actor_freeze_updates = int(actor_freeze_updates)
        # PER-MODE Q NORMALISATION of the actor loss (see _gated_actor_update).
        # Scales are EMAs of each critic's masked batch std, initialised from
        # the first batch that carries the mode; floored so an untrained critic
        # (std ~ 0) cannot blow the gradient up.
        self.actor_q_norm = bool(actor_q_norm)
        self.actor_q_norm_tau = float(actor_q_norm_tau)
        # QNORM_CRITIC_ALPHA (opt-in, default off). With ACTOR_Q_NORM the
        # actor's -Q term is divided by the per-mode value spread s, so its
        # effective entropy temperature is alpha * s -- while the critics'
        # backups soft-maximise against plain alpha. With this flag on, each
        # critic's entropy term is scaled by the SAME s its mode's actor rows
        # divide by, so the two sides of the TD equation agree again. Off by
        # default because it changes the critic target: it is a separate
        # experiment, tested one flag at a time (TQC_VALUE_CEILING first).
        self.qnorm_critic_alpha = bool(qnorm_critic_alpha)
        self._q_scale_g = None
        self._q_scale_e = None
        self._updates = 0

        # THE SECOND CRITIC IS BUILT BY CONSTRUCTING A SIBLING LEARNER AND
        # TAKING ITS CRITIC. Replicating the network construction here would
        # duplicate the parent's encoder selection, PixelMultiplexer wiring and
        # ensemble sizing, and would silently drift the moment the parent
        # changed. Borrowing the parent's own constructor makes drift
        # impossible. A different seed gives independent initialisation; the
        # sibling's actor is discarded, which costs one throwaway MLP at start-up
        # and nothing thereafter.
        sib_kwargs = dict(kwargs)
        if init_temperature_e is not None:
            sib_kwargs["init_temperature"] = init_temperature_e
        sibling = TQCLearner(seed + 10_000, strip_priv(observations), actions,
                             **sib_kwargs)
        self._critic_e = sibling._critic
        self._target_critic_e_params = copy.deepcopy(sibling._critic.params)
        self._temp_e = sibling._temp
        del sibling

        # THE PRIVILEGED GOAL CRITIC, built the same way from a sibling that
        # DOES see the lane block. Its temperature is the goal one (self._temp).
        self._critic_p = None
        self._target_critic_p_params = None
        if self.asymmetric:
            sib_p = TQCLearner(seed + 20_000, observations, actions, **dict(kwargs))
            self._critic_p = sib_p._critic
            self._target_critic_p_params = copy.deepcopy(sib_p._critic.params)
            del sib_p
            print("[Gated] ASYMMETRIC goal critic: a PRIVILEGED goal critic "
                  f"(sees '{PRIV_KEY}') drives the actor on goal rows; the "
                  "image-only goal critic trains on the same batches and is "
                  "the distance the map builder reads (agent._critic). The "
                  f"actor and the explore critic never see '{PRIV_KEY}'.")

        bounds = ("unbounded" if self.value_floor is None
                  and self.value_ceiling is None
                  else f"[{self.value_floor}, {self.value_ceiling}]")
        print(f"[Gated] two critics: discount={self.discount} "
              f"discount_e={self.discount_e} "
              f"target_entropy={self.target_entropy} "
              f"target_entropy_e={self.target_entropy_e} "
              f"explore_actor_warmup={self.explore_actor_warmup} "
              f"actor_freeze_updates={self.actor_freeze_updates} "
              f"actor_q_norm={self.actor_q_norm} "
              f"qnorm_critic_alpha={self.qnorm_critic_alpha}")
        print(f"[Gated] goal critic target clip {bounds}; "
              f"explore critic UNCLIPPED (its range is its own, not the goal arm's)")
        if self.discount >= 1.0 and self.value_floor is None:
            print("[Gated] WARNING: discount=1.0 with NO value floor. V(s) = "
                  "-c + V(s') has no bounded fixed point on a state the agent "
                  "never escapes, and a truncated episode bootstraps rather "
                  "than terminating. Set TQC_VALUE_FLOOR.")

    def update(self, batch, utd_ratio: int = None,
               enable_update_temperature: bool = True,
               output_range=None) -> Dict[str, float]:
        # utd_ratio is accepted for signature parity and deliberately ignored:
        # the mode-split buffer concatenates goal rows FIRST, so slicing the
        # batch would hand whole slices a single mode and give the other
        # critic a zero-row gradient for that slice.
        explore_actor_w = (0.0 if self._updates < self.explore_actor_warmup
                           else 1.0)
        (rng, actor, critic, target_params, temp,
         critic_e, target_e_params, temp_e, info,
         critic_p, target_p_params) = _gated_update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params,
            self._temp, self._critic_e, self._target_critic_e_params,
            self._temp_e, batch,
            self.discount, self.discount_e, self.tau, explore_actor_w,
            self.target_entropy, self.target_entropy_e, self.backup_entropy,
            self.top_quantiles_to_drop, self.kappa,
            value_floor=self.value_floor, value_ceiling=self.value_ceiling,
            enable_update_temperature=enable_update_temperature,
            augument=self.augument, share_encoder=self.share_encoder,
            q_scale_g=(float(self._q_scale_g) if (self.actor_q_norm and self._q_scale_g) else 1.0),
            q_scale_e=(float(self._q_scale_e) if (self.actor_q_norm and self._q_scale_e) else 1.0),
            # QNORM_CRITIC_ALPHA: the critics' entropy backups see the same
            # per-mode spread the actor divides Q by. Reuses the running
            # scales (set only after the first batch carrying each mode) and
            # stays 1.0 -- byte-identical -- unless BOTH flags are on.
            alpha_scale_g=(float(self._q_scale_g)
                           if (self.actor_q_norm and self.qnorm_critic_alpha
                               and self._q_scale_g) else 1.0),
            alpha_scale_e=(float(self._q_scale_e)
                           if (self.actor_q_norm and self.qnorm_critic_alpha
                               and self._q_scale_e) else 1.0),
            critic_p=self._critic_p,
            target_critic_p_params=self._target_critic_p_params,
            bc_weight=self.bc_weight, bc_scale=self._bc_scale,
            # Set by the trainer after every [GatedEp] block. A plain float, so
            # it is traced, not static: a new value each update recompiles nothing.
            lagrange_mu=float(getattr(self, "lagrange_mu", 0.0)))

        self._rng = rng
        frozen = self._updates < self.actor_freeze_updates
        if not frozen:
            # the jitted step always computes an actor update; while frozen it
            # is simply discarded, so the critics train against a fixed actor
            self._actor = actor
            self._temp = temp
            self._temp_e = temp_e
        self._critic = critic
        self._target_critic_params = target_params
        self._critic_e = critic_e
        self._target_critic_e_params = target_e_params
        if critic_p is not None:
            self._critic_p = critic_p
            self._target_critic_p_params = target_p_params
        self._updates += 1
        info = dict(info)
        info["actor_frozen"] = float(frozen)
        if self.actor_q_norm:
            sg, se = float(info["q_std_goal"]), float(info["q_std_explore"])
            t = self.actor_q_norm_tau
            if sg > 0:
                self._q_scale_g = max(1e-3, sg if self._q_scale_g is None
                                      else (1 - t) * self._q_scale_g + t * sg)
            if se > 0:
                self._q_scale_e = max(1e-3, se if self._q_scale_e is None
                                      else (1 - t) * self._q_scale_e + t * se)
            info["q_scale_goal"] = float(self._q_scale_g or 1.0)
            info["q_scale_explore"] = float(self._q_scale_e or 1.0)
        return info

    # -- the actor's entry points: never the privileged block --------------
    # jaxrl2's Agent passes whatever it is given straight to the actor, whose
    # network has no input layer for the lane block. Every method that reaches
    # the actor (or the image-only critic) strips it first, so a caller may
    # hand over the env's full observation.
    def eval_actions(self, observations):
        return super().eval_actions(strip_priv(observations))

    def sample_actions(self, observations):
        return super().sample_actions(strip_priv(observations))

    def act(self, observations):
        return super().act(strip_priv(observations))

    def action_dist(self, batch):
        return super().action_dist(strip_priv(batch))

    def eval_log_probs(self, batch):
        return super().eval_log_probs(strip_priv(batch))

    def extract_features(self, batch):
        return super().extract_features(strip_priv(batch))

    def ensemble_uncertainty(self, batch, action):
        return super().ensemble_uncertainty(strip_priv(batch), action)
