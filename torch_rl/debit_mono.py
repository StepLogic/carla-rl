#!/usr/bin/env python3
"""Load DEBiT's PRETRAINED monocular branch into our ResNet.

User 2026-10-14: "I want debit Mono + binocular setup", then "use group norm for
resnet". Both branches of DEBiT's perception, not just the binocular one.

THE WEIGHTS FIT BECAUSE THE ARCHITECTURE IS ALREADY THEIRS. torch_rl/resnet_mono
is DEBiT's monocular branch reimplemented -- a half-width ResNet-18 with
GroupNorm -- and the released checkpoint's monocular encoder is 4.14 M
parameters against our 4.14 M, with 60 of our 65 tensors matching by name and
shape once the `resnet.` prefix is stripped. GroupNorm is what makes this clean:
a BatchNorm trunk would carry running statistics that our module has no place
for, and it is also the right normalisation for a ResNet trained inside RL,
where batches are small and highly correlated.

WHAT DOES NOT TRANSFER, AND WHY THAT IS FINE:
  * the HEAD. Theirs is a single fc; ours is Vc1Compression (conv + fc), the same
    compression the rest of this learner uses. It stays randomly initialised and
    trains -- it is the part that has to learn what THIS policy needs from the
    features, and it is small.
  * `running_mean_and_var`. Habitat's ResNet normalises its input by a RUNNING
    estimate rather than fixed constants, so the pretrained weights were trained
    behind that distribution, not behind the ImageNet constants our module
    applies. Loading the weights and ignoring their statistics would feed them
    something they were never calibrated for, so the stored mean and variance are
    copied into our mean/std buffers. This is the one place where "it loaded
    cleanly" would not have meant "it is correct".

FINE-TUNED, NOT FROZEN. The binocular branch is frozen because its output is then
a fixed function that the env can precompute; the monocular branch has no such
role, its features came from Habitat indoor scenes, and CARLA is roads. Freezing
it would leave the compression head as the only trainable perception in the arm.
"""
import os
from typing import Dict

import torch

MONO_PREFIX = "actor_critic.net.monocular_encoder."
TRUNK_PREFIX = "resnet."


def debit_mono_tensors(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """The checkpoint's monocular tensors, `resnet.` stripped."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    scripts = os.path.join(here, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from inspect_debit_ckpt import load_tensors
    all_t = load_tensors(ckpt_path)
    mono = {k[len(MONO_PREFIX):]: v for k, v in all_t.items()
            if k.startswith(MONO_PREFIX)}
    if not mono:
        raise SystemExit(
            f"{ckpt_path} has no tensors under {MONO_PREFIX!r}; this is not a "
            "released DEBiT agent checkpoint.")
    return {(k[len(TRUNK_PREFIX):] if k.startswith(TRUNK_PREFIX) else k): v
            for k, v in mono.items()}


def load_debit_mono(encoder, ckpt_path: str, adopt_input_stats: bool = True,
                    min_loaded: int = 50) -> Dict[str, object]:
    """Load the pretrained trunk into `encoder` (a MonoResNetEncoder).

    Returns a report. Raises if suspiciously little loaded: a silent near-miss
    here gives a randomly-initialised trunk wearing a pretrained label, which no
    downstream number would reveal.
    """
    theirs = debit_mono_tensors(ckpt_path)
    ours = encoder.state_dict()
    take, skipped = {}, []
    for k, v in theirs.items():
        if k in ours and tuple(ours[k].shape) == tuple(v.shape):
            take[k] = v
        else:
            skipped.append(k)
    if len(take) < int(min_loaded):
        raise SystemExit(
            f"only {len(take)} of {len(theirs)} DEBiT monocular tensors matched "
            f"(expected >= {min_loaded}). Loading this would leave a random "
            "trunk described as pretrained.")
    missing = [k for k in ours if k not in take]
    encoder.load_state_dict(take, strict=False)

    stats = None
    if adopt_input_stats:
        mean = theirs.get("running_mean_and_var._mean")
        var = theirs.get("running_mean_and_var._var")
        if mean is not None and var is not None:
            with torch.no_grad():
                # Their pixels are /255 like ours, so the statistics are directly
                # comparable; std is sqrt(var), floored so a degenerate channel
                # cannot divide by zero.
                encoder.mean.copy_(mean.to(encoder.mean.device).reshape_as(encoder.mean))
                encoder.std.copy_(var.to(encoder.std.device).reshape_as(encoder.std)
                                  .clamp_min(1e-8).sqrt())
            stats = (float(mean.mean()), float(var.mean()))
    return {"loaded": len(take), "skipped": skipped, "missing": missing,
            "input_stats": stats}


def describe_load(report: Dict[str, object]) -> str:
    s = (f"DEBiT monocular: {report['loaded']} pretrained tensors loaded, "
         f"{len(report['missing'])} left at init "
         f"({', '.join(sorted({m.split('.')[0] for m in report['missing']}))})")
    if report.get("input_stats"):
        m, v = report["input_stats"]
        s += f"; input stats adopted (mean {m:.3f}, var {v:.3f})"
    else:
        s += "; ImageNet input constants kept"
    return s + "; FINE-TUNED"
