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
                  gamma = 1.0, so its Q IS the cost-to-go in steps. The explore
                  arm pays a lane-progress bonus at gamma = 0.99. The discounts
                  genuinely differ and the returns differ in SIGN, so one critic
                  could not serve both even at equal scale. This is the Habitat
                  peer's rationale, and under this reward it does transfer.

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
is clipped to that window. The clip is PER CRITIC: the explore critic's targets
are positive and must never see it.

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

from vae_nav.quantile_per_sample import (masked_mean,
                                         quantile_huber_loss_per_sample)

# Must match vae_nav.gated_reward and vae_nav.vae_goal_wrapper.
GOAL, EXPLORE = 1, 0
MODE_KEY = "mode"


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
                          kappa, mask, value_floor=None, value_ceiling=None):
    """One critic's update, restricted to the rows `mask` selects.

    value_floor / value_ceiling clip this critic's TARGET only, and are applied
    PER CRITIC. Under the SORB minimum-time reward the goal critic runs at
    gamma = 1.0, where V(s) = -c + V(s') has no bounded fixed point on a state
    the agent never escapes; the bound makes the return range explicit
    ([-step_cost * max_steps, 0]) so a target outside it is unrealisable rather
    than merely unlikely.

    THEY CANNOT COME FROM THE GLOBAL TQC_TARGET_MIN/MAX. Those are read at
    import in critic_updater and baked into every trace, so they would also clip
    the EXPLORE critic -- whose targets are POSITIVE -- to zero, silently
    destroying it. The explore critic passes None here and is left unclipped.
    """
    target_critic = critic.replace(params=target_critic_params)
    dist = actor.apply_fn({"params": actor.params}, batch["next_observations"])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_quantiles = target_critic.apply_fn(
        {"params": target_critic_params}, batch["next_observations"], next_actions)
    alpha = temp.apply_fn({"params": temp.params}) if backup_entropy else 0.0

    # THE TARGET IS COMPUTED FOR EVERY ROW, at this critic's own discount, and
    # the wrong-mode rows are discarded by the mask below. That wastes compute
    # on half the batch and is deliberate: the alternative is dynamic control
    # flow inside a jitted function, whose shapes the compiler cannot fix.
    target = tqc_target(
        next_quantiles,
        jnp.nan_to_num(batch["rewards"]),
        batch["masks"],
        discount=discount,
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
                      "td_errors": masked_mean(td, mask)}

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    return critic.apply_gradients(grads=grads), info


def _gated_actor_update(key, actor, critic, critic_e, temp, temp_e, batch,
                        m_goal, explore_actor_w):
    """One actor, choosing its critic and its temperature PER ROW."""
    alpha_g = temp.apply_fn({"params": temp.params})
    alpha_e = temp_e.apply_fn({"params": temp_e.params})
    is_goal = m_goal > 0.5
    m_explore = 1.0 - m_goal

    def actor_loss_fn(params):
        dist = actor.apply_fn({"params": params}, batch["observations"])
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        # Both critics are evaluated on the whole batch and half of each result
        # is thrown away. Same trade as above: a fixed shape the JIT can handle.
        q_g = critic.apply_fn({"params": critic.params},
                              batch["observations"], actions).mean(axis=(0, 2))
        q_e = critic_e.apply_fn({"params": critic_e.params},
                                batch["observations"], actions).mean(axis=(0, 2))
        q = jnp.where(is_goal, q_g, q_e)
        alpha = jnp.where(is_goal, alpha_g, alpha_e)
        per_row = alpha * log_probs - q
        # explore_actor_w damps the explore-mode gradient during warm-up. After
        # a warm start the explore critic is stale, and an actor following a
        # stale critic moves away from a policy that was already good.
        loss = jnp.where(is_goal, per_row, explore_actor_w * per_row).mean()
        return loss, {
            "actor_loss": loss,
            "entropy": -log_probs.mean(),
            "entropy_goal": masked_mean(-log_probs, m_goal),
            "entropy_explore": masked_mean(-log_probs, m_explore),
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
                      share_encoder=True):
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

    m_goal = _mode_mask(batch)
    m_explore = 1.0 - m_goal

    rng, key_g = jax.random.split(rng)
    # GOAL CRITIC: bounded, because under the SORB reward it runs at gamma=1.0.
    new_critic, info_g = _masked_critic_update(
        key_g, actor, critic, target_critic_params, temp, batch, discount,
        backup_entropy, top_quantiles_to_drop, kappa, m_goal,
        value_floor=value_floor, value_ceiling=value_ceiling)

    # EXPLORE CRITIC: UNBOUNDED, deliberately. Its targets are POSITIVE and its
    # discount is 0.99, so it has a bounded fixed point of its own and the goal
    # critic's [-N, 0] window would clip every one of its targets to zero.
    rng, key_e = jax.random.split(rng)
    new_critic_e, info_e = _masked_critic_update(
        key_e, actor, critic_e, target_critic_e_params, temp_e, batch,
        discount_e, backup_entropy, top_quantiles_to_drop, kappa, m_explore)

    new_target_params = soft_target_update(
        new_critic.params, target_critic_params, tau)
    new_target_e_params = soft_target_update(
        new_critic_e.params, target_critic_e_params, tau)

    rng, key_a = jax.random.split(rng)
    new_actor, info_a = _gated_actor_update(
        key_a, actor, new_critic, new_critic_e, temp, temp_e, batch,
        m_goal, explore_actor_w)

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
        "critic_loss_e": info_e["critic_loss"], "q_e": info_e["q"],
        "td_errors_e": info_e["td_errors"],
        "frac_goal": m_goal.mean(),
    }
    return (rng, new_actor, new_critic, new_target_params, new_temp,
            new_critic_e, new_target_e_params, new_temp_e, info)


class GatedTQCLearner(TQCLearner):
    """TQCLearner with a second critic and temperature for the explore mode."""

    def __init__(self, seed, observations, actions, discount_e: float = 0.99,
                 init_temperature_e: Optional[float] = None,
                 target_entropy_e: Optional[float] = None,
                 explore_actor_warmup: int = 0,
                 value_floor: Optional[float] = None,
                 value_ceiling: Optional[float] = None, **kwargs):
        super().__init__(seed, observations, actions, **kwargs)

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
        sibling = TQCLearner(seed + 10_000, observations, actions, **sib_kwargs)
        self._critic_e = sibling._critic
        self._target_critic_e_params = copy.deepcopy(sibling._critic.params)
        self._temp_e = sibling._temp
        del sibling

        bounds = ("unbounded" if self.value_floor is None
                  and self.value_ceiling is None
                  else f"[{self.value_floor}, {self.value_ceiling}]")
        print(f"[Gated] two critics: discount={self.discount} "
              f"discount_e={self.discount_e} "
              f"target_entropy={self.target_entropy} "
              f"target_entropy_e={self.target_entropy_e} "
              f"explore_actor_warmup={self.explore_actor_warmup}")
        print(f"[Gated] goal critic target clip {bounds}; "
              f"explore critic UNCLIPPED (its targets are positive)")
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
         critic_e, target_e_params, temp_e, info) = _gated_update_jit(
            self._rng, self._actor, self._critic, self._target_critic_params,
            self._temp, self._critic_e, self._target_critic_e_params,
            self._temp_e, batch,
            self.discount, self.discount_e, self.tau, explore_actor_w,
            self.target_entropy, self.target_entropy_e, self.backup_entropy,
            self.top_quantiles_to_drop, self.kappa,
            value_floor=self.value_floor, value_ceiling=self.value_ceiling,
            enable_update_temperature=enable_update_temperature,
            augument=self.augument, share_encoder=self.share_encoder)

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_params
        self._temp = temp
        self._critic_e = critic_e
        self._target_critic_e_params = target_e_params
        self._temp_e = temp_e
        self._updates += 1
        return info
