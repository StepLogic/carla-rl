"""The critic's projection head, pretrained by imitation on the demos and frozen
(user 2026-11-09: "pretrain projection layer with imitation learning using demo data
like c0 before - freeze and start town05_town01 SAC train").

The head is DrQ's critic input head (nets.QuantileCritics proj_dim): tanh(LayerNorm(
x W + b)) on the critic's own input x = [h | state cols]. Here ONE head learns, with
c0's cloning objective, to carry what the expert does in the state: an actor-sized
decoder (the actor's MLP, mean and log-std) reads the head's output and the loss is
bc_trainer.bc_loss's -- the pre-tanh Gaussian NLL of the expert's label action, label
clipped to +-0.995 -- over the valid rows of each chunk. The trunk (encoder + GRU)
only provides h, without gradient. Which head is kept is `select`: "final" (the
default; user 2026-11-09, "Restart with the 10k head") is the head after all updates,
a fixed schedule as c0's training was; "nll" early-stops on the held-out NLL among
trained heads. Held-out NLL and action error are logged either way (on the cluster
demos the NLL mostly scored confidence -- c0's own actor 56.8 -- and the head it
early-stopped had the worst action error, 0.357, against the 10k head's 0.286 and
c0's 0.304; the untrained head's ~0 predictions scored 0.319). It is then copied into
every critic of the goal ensemble and its target, and frozen there: as in DrQ, where
all critics share the encoder's output, the critics differ only in the MLPs SAC trains
on top of it.
"""
import copy
import math
import time

import torch
import torch.nn as nn

from torch_rl.nets import LOG_STD_MAX, LOG_STD_MIN, _mlp

_HEAD = ("proj_w", "proj_b", "proj_ln_w", "proj_ln_b")


class ProjectionHead(nn.Module):
    """One critic's input head, parameter for parameter: (D, P) weight, P bias,
    P LayerNorm gain and shift -- QuantileCritics.project for a single critic."""

    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        w = torch.empty(int(in_dim), int(proj_dim))
        nn.init.kaiming_uniform_(w.T, a=math.sqrt(5))      # read as (d_in, d_out), as there
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(int(proj_dim)))
        self.ln_w = nn.Parameter(torch.ones(int(proj_dim)))
        self.ln_b = nn.Parameter(torch.zeros(int(proj_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.w + self.b
        z = nn.functional.layer_norm(z, (z.shape[-1],)) * self.ln_w + self.ln_b
        return torch.tanh(z)


def _critic_inputs(agent, chunk):
    """(x, labels, valid) on the chunk's trained rows: x = [h | state cols], the
    goal critic's input, as _update_impl and bc_loss align them."""
    cfg, dev = agent.cfg, agent.device
    obs, keep, actions, valid = (chunk[k].to(device=dev, dtype=torch.float32)
                                 for k in ("obs", "keep", "actions", "valid"))
    goal = chunk["goal"].to(device=dev, dtype=torch.float32) if cfg.goal_tokens else None
    K = int(cfg.burn_in)
    B, T = valid.shape
    N = B * T
    with torch.no_grad():
        s_all = agent._roll(agent.compress, agent.act_embed, agent.core, obs, keep, goal)
        s = s_all[:, :-1].reshape(N, cfg.hidden).float()
        st = agent._state(obs[:, K:-1]).reshape(N, cfg.state_dim).float()
        x = torch.cat([s, st.index_select(-1, agent.goal_cols)], dim=-1)
    return x, actions.reshape(N, -1), valid.reshape(N) > 0


def _nll_mae(mean, log_std, a, v, clip=0.995):
    """bc_trainer.bc_loss's loss and deterministic error, on precomputed outputs."""
    u = torch.atanh(a.clamp(-clip, clip))
    nll = (0.5 * ((u - mean) / log_std.exp()) ** 2 + log_std).sum(-1)
    n = v.sum().clamp_min(1)
    loss = torch.where(v, nll, torch.zeros_like(nll)).sum() / n
    mae = torch.where(v, (torch.tanh(mean) - a).abs().mean(-1), torch.zeros_like(nll)).sum() / n
    return loss, mae


def _decode(decoder, z):
    mean, log_std = decoder(z).chunk(2, dim=-1)
    return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)


def pretrain_critic_projection(agent, sample_train, val_chunk, *, updates, lr=3e-4,
                               eval_every=500, freeze=True, log=print, seed=0,
                               select="final"):
    """Train one projection head by c0's cloning loss and keep one (`select`):
    "final" = the head after all `updates` (user 2026-11-09, "Restart with the 10k
    head": on held-out data the NLL mostly scores confidence -- c0's own actor gets
    56.8 -- and the early-stopped head had the worst action error, 0.357, against the
    10k head's 0.286 and c0's 0.304), or "nll" = early stopping on the held-out NLL
    among trained heads. The held-out `val_chunk` (one chunk or a list) is measured
    either way. The kept head is copied into every goal critic and its target and
    frozen there (`freeze`). `sample_train()` returns a labelled chunk (the bc_trainer
    buffers' format). Returns stats: init / kept ("best_*") / final validation NLL and
    MAE, c0's own actor on the same rows, the number of rows, the kept update, the
    rule and the history."""
    if select not in ("final", "nll"):
        raise ValueError(f"select={select!r}: expected 'final' or 'nll'")
    C = agent.critics_g
    P = int(getattr(C, "proj_dim", 0) or 0)
    if P <= 0:
        raise ValueError("the critic has no projection head: set CRITIC_PROJ_DIM > 0")
    dev = agent.device
    torch.manual_seed(int(seed))
    head = ProjectionHead(C.proj_w.shape[1], P).to(dev)
    A = int(agent.cfg.action_dim)
    decoder = _mlp(P, tuple(agent.cfg.mlp), 2 * A).to(dev)
    opt = torch.optim.Adam(list(head.parameters()) + list(decoder.parameters()), lr=float(lr))

    chunks = val_chunk if isinstance(val_chunk, (list, tuple)) else [val_chunk]
    parts = [_critic_inputs(agent, ch) for ch in chunks]
    xv = torch.cat([p[0] for p in parts])
    av = torch.cat([p[1] for p in parts])
    vv = torch.cat([p[2] for p in parts])
    actor_in = torch.cat([torch.cat([p[0][:, :agent.cfg.hidden], _state_cols(agent, ch)], -1)
                          for p, ch in zip(parts, chunks)])
    del parts

    def val():
        with torch.no_grad():
            nll, mae = _nll_mae(*_decode(decoder, head(xv)), av, vv)
        return float(nll), float(mae)

    with torch.no_grad():
        # c0's own actor on the same rows: what the full input gives (reference).
        am, als = agent._actor_forward(actor_in)
        anll, amae = _nll_mae(am, als, av, vv)
    del actor_in
    init_nll, init_mae = val()
    best = None              # trained heads only: the untrained one is never kept
    history, t0, window = [], time.time(), []
    for step in range(1, int(updates) + 1):
        x, a, v = _critic_inputs(agent, sample_train())
        loss, mae = _nll_mae(*_decode(decoder, head(x)), a, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        window.append((float(loss.detach()), float(mae.detach())))
        if step % int(eval_every) == 0 or step == int(updates):
            vn, vm = val()
            tn = sum(w[0] for w in window) / len(window)
            tm = sum(w[1] for w in window) / len(window)
            history.append({"step": step, "train_nll": tn, "train_mae": tm,
                            "val_nll": vn, "val_mae": vm})
            # EARLY STOPPING on the validation NLL, trained heads only. (Action error
            # cannot choose: on the cluster data the untrained head's ~0 predictions
            # scored 0.310, better than every trained head up to 2500 updates, while
            # the NLL fell 0.75 -> -1.32 at 1000 updates and rose after: overfitting.)
            if (select == "final" and step == int(updates)) or (
                    select == "nll" and (best is None or vn < best["val_nll"])):
                best = {"val_nll": vn, "val_mae": vm, "step": step,
                        "head": copy.deepcopy(head.state_dict()),
                        "decoder": copy.deepcopy(decoder.state_dict())}
            kept = ("keeping the final head" if select == "final" else
                    f"best val_nll {best['val_nll']:.4f} at {best['step']}")
            log(f"[PROJ] step={step} train_nll={tn:.4f} val_nll={vn:.4f} val_mae={vm:.4f} "
                f"train_mae={tm:.4f} ({kept}; "
                f"{len(window) / (time.time() - t0):.2f} upd/s)")
            window, t0 = [], time.time()
    head.load_state_dict(best["head"])
    decoder.load_state_dict(best["decoder"])       # the pair the best NLL was measured on
    final_nll, final_mae = val()
    with torch.no_grad():
        for crit in (agent.critics_g, agent.critics_g_targ):
            for name, src in zip(_HEAD, (head.w, head.b, head.ln_w, head.ln_b)):
                dst = getattr(crit, name)
                dst.copy_(src.detach()[None].expand_as(dst))
                if freeze:
                    dst.requires_grad_(False)
    return {"init_val_nll": init_nll, "init_val_mae": init_mae,
            "best_val_nll": best["val_nll"], "best_val_mae": best["val_mae"],
            "best_step": best["step"], "final_val_nll": final_nll, "final_val_mae": final_mae,
            "actor_val_nll": float(anll), "actor_val_mae": float(amae),
            "val_rows": int(vv.sum()), "frozen": bool(freeze), "proj_dim": P,
            "select": select, "history": history}


def _state_cols(agent, chunk):
    """The full state block of the chunk's trained rows (the actor's input after h)."""
    cfg, dev = agent.cfg, agent.device
    obs = chunk["obs"].to(device=dev, dtype=torch.float32)
    K = int(cfg.burn_in)
    B, T = chunk["valid"].shape
    with torch.no_grad():
        return agent._state(obs[:, K:-1]).reshape(B * T, cfg.state_dim).float()
