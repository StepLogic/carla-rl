"""Save and restore the TQC policy's WEIGHTS.

WHY THIS EXISTS. The trainer wrote a `checkpoint_{step}.pkl` per evaluation
containing step, identity, eval_success, per-town rates, episode counts and HER
stats -- everything ABOUT the run and nothing OF it. The networks were never
serialised, so a finished run left no policy behind: it could not be replayed,
watched, evaluated later, or compared against a subsequent run. Fifty thousand
steps of training existed only inside a process that had already exited.

This closes that. `save_agent` folds the actor/critic/temperature parameters
into the same pickle the trainer already writes, so nothing about the existing
checkpoint layout changes -- the files simply stop being empty of the model.

WHAT IS SAVED. The `params` pytrees of the three TrainStates, converted to
numpy so the pickle does not carry live device buffers (which fail to unpickle
on a machine with a different accelerator, and pin GPU memory when they do
not). Optimiser state is deliberately NOT saved: these checkpoints exist to
replay and evaluate a policy, not to resume an interrupted run, and optimiser
moments would roughly triple the file for a use we do not have.

RESTORE IS BY `.replace()`, NOT RECONSTRUCTION. The agent is rebuilt through
the trainer's own `build_agent` -- identical architecture, identical action
space -- and only the leaves are swapped in. A separately-written network
definition here would be free to drift from the one that trained, and the
mismatch would surface as plausible-looking but wrong actions rather than an
error.
"""
import numpy as np


def _to_numpy(tree):
    import jax
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def agent_state(agent):
    """The parameter pytrees, as numpy. Safe to pickle anywhere.

    A GATED agent carries three tensors a plain one does not: the explore
    critic, its target, and the explore temperature. They are written only when
    present, so a plain checkpoint keeps exactly the keys it has always had and
    stays readable by everything that reads one today.
    """
    state = {"actor": _to_numpy(agent._actor.params),
             "critic": _to_numpy(agent._critic.params)}
    temp = getattr(agent, "_temp", None)
    if temp is not None:
        state["temp"] = _to_numpy(temp.params)

    critic_e = getattr(agent, "_critic_e", None)
    if critic_e is not None:
        state["critic_e"] = _to_numpy(critic_e.params)
        # The explore target is saved explicitly rather than derived from the
        # online critic on load. The two diverge by design between soft updates,
        # and a resume that silently reset the target to the online parameters
        # would restart the smoothing this agent depends on.
        target_e = getattr(agent, "_target_critic_e_params", None)
        state["target_critic_e"] = _to_numpy(
            target_e if target_e is not None else critic_e.params)
    temp_e = getattr(agent, "_temp_e", None)
    if temp_e is not None:
        state["temp_e"] = _to_numpy(temp_e.params)
    # ASYMMETRIC (OBS_PRIV): the privileged goal critic and its target. Only
    # written when present, so every other checkpoint keeps its keys.
    critic_p = getattr(agent, "_critic_p", None)
    if critic_p is not None:
        state["critic_p"] = _to_numpy(critic_p.params)
        state["target_critic_p"] = _to_numpy(agent._target_critic_p_params)
    return state


def load_agent_state(agent, state):
    """Swap saved leaves into an agent built by the trainer's `build_agent`.

    Structure is checked before assignment: an actor trained against a
    different observation space unpickles perfectly happily and then produces
    confident nonsense, which is far harder to notice than a load error.
    """
    import jax

    def _swap(train_state, saved, name):
        want = jax.tree_util.tree_structure(train_state.params)
        got = jax.tree_util.tree_structure(saved)
        if want != got:
            raise SystemExit(
                f"checkpoint '{name}' does not match the agent built here.\n"
                f"  expected: {want}\n  saved:    {got}\n"
                "The observation space or network shape changed since this "
                "checkpoint was written. Loading it anyway would give a policy "
                "that acts confidently on a layout it never trained on.")
        return train_state.replace(params=saved)

    agent._actor = _swap(agent._actor, state["actor"], "actor")
    agent._critic = _swap(agent._critic, state["critic"], "critic")
    if "temp" in state and getattr(agent, "_temp", None) is not None:
        agent._temp = _swap(agent._temp, state["temp"], "temp")
    # The target critic rides along with the online one: an eval-only load has
    # no use for a separate target, and leaving it at its random init would
    # make any later resume start from a mismatched pair.
    if hasattr(agent, "_target_critic_params"):
        agent._target_critic_params = state["critic"]
    elif hasattr(agent, "_target_critic"):
        agent._target_critic = agent._target_critic.replace(params=state["critic"])

    # ---- the gated half -------------------------------------------------
    #
    # Three cases, and the asymmetry is deliberate.
    #
    # gated -> plain: the extra keys are IGNORED, not an error. Evaluation
    #   harnesses build a plain agent, and refusing here would mean a gated
    #   checkpoint could never be evaluated by them.
    # gated -> gated: full restore, both critics and both targets.
    # plain -> gated: WARM START. The explore critic has no counterpart and is
    #   left at its initialisation. It must NOT be filled from the goal critic:
    #   that critic is fitted to a return an order of magnitude larger, so
    #   seeding the explore critic with it starts training from a confidently
    #   wrong value function rather than an uninformed one.
    gated_agent = getattr(agent, "_critic_e", None) is not None
    gated_state = "critic_e" in state
    if gated_agent and gated_state:
        agent._critic_e = _swap(agent._critic_e, state["critic_e"], "critic_e")
        agent._target_critic_e_params = state.get("target_critic_e",
                                                  state["critic_e"])
        if "temp_e" in state and getattr(agent, "_temp_e", None) is not None:
            agent._temp_e = _swap(agent._temp_e, state["temp_e"], "temp_e")
    elif gated_agent and not gated_state:
        print("[policy_io] WARM START: this checkpoint has no explore critic, "
              "so the actor and goal critic were restored and the EXPLORE "
              "CRITIC was left at its initialisation. Expect its value "
              "estimates to be meaningless until it has trained; damp the "
              "explore-mode actor gradient (explore_actor_warmup) or the actor "
              "will follow a critic that knows nothing.")
    # ASYMMETRIC: restore the privileged goal critic when both sides have one.
    # A privileged agent loading a checkpoint WITHOUT it is refused: resuming
    # would hand the actor to a randomly initialised critic.
    if getattr(agent, "_critic_p", None) is not None:
        if "critic_p" not in state:
            raise SystemExit(
                "this agent has a privileged goal critic (OBS_PRIV=1) but the "
                "checkpoint has none; the actor would follow an untrained "
                "critic. Start fresh or load a checkpoint from an OBS_PRIV run.")
        agent._critic_p = _swap(agent._critic_p, state["critic_p"], "critic_p")
        agent._target_critic_p_params = state.get("target_critic_p",
                                                  state["critic_p"])
    return agent


def param_fingerprint(state):
    """A cheap scalar summary, printed on save and on load.

    Two runs that print the same fingerprint loaded the same weights. This is
    what catches "the checkpoint restored fine and the policy is still random"
    -- the failure that has no exception attached to it.
    """
    import jax
    leaves = jax.tree_util.tree_leaves(state["actor"])
    total = sum(int(np.asarray(x).size) for x in leaves)
    checksum = float(sum(float(np.asarray(x).sum()) for x in leaves))
    return {"actor_params": total, "actor_checksum": round(checksum, 4)}
