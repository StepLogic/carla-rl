"""Appearance novelty: how like anything already in the buffer is this view?

Complements the geometric visit ring rather than replacing it. The two answer
different questions and fail in different places:

    visit ring        "have I BEEN at that coordinate"   directional, 16 slots
    image novelty     "have I SEEN this view"            appearance, scalar

WHY BOTH. The peer's v1 ring is computed from TRUE SIMULATOR POSE against a
cell grid -- privileged information the agent could not have on a real vehicle,
and this project has already rejected privileged map data once on exactly that
ground. Image similarity needs no pose at all: it asks whether the current
frame resembles a stored one. So the ring is the stronger signal where pose is
available, and the novelty scalar is the one that would survive on hardware.
Keeping both lets an ablation say which is carrying the behaviour.

It is also nearly free. The MobileNetV3 backend already computes a 384-d
embedding of this frame for the policy's observation, so novelty costs one
matrix-vector product against the buffer and no extra forward pass.

SIMILARITY IS COSINE ON L2-NORMALISED FEATURES. Raw dot products on
un-normalised conv features track brightness and contrast more than content --
a bright wall and a bright road look similar by magnitude alone. Normalising
first is what makes the comparison about direction in feature space.

BUFFER ADMISSION IS THRESHOLDED, NOT EVERY-STEP. Storing every frame makes the
buffer a near-duplicate pile: consecutive frames at 0.283 m apart are almost
identical, so max-similarity saturates at ~1.0 within a few steps and the
signal dies. Admitting a frame only when it is sufficiently UNLIKE everything
stored keeps the buffer a set of distinct places, which is the same reason
SPTM keeps keyframes rather than all frames.
"""
import os

import numpy as np

# Above this cosine similarity to something already stored, a frame is "seen".
#
# 0.9 IS CALIBRATED ON SYNTHETIC STRUCTURED IMAGES, NOT ON CARLA FRAMES.
# MEASURED with the MobileNetV3 backend: same place 0.966, different place
# 0.691, so 0.9 separates them cleanly. But road frames are far more
# self-similar than those stimuli -- two stretches of the same town look alike
# -- so the different-place figure will rise on real data and 0.9 may admit
# almost nothing. CHECK admit_rate FROM stats() ON THE FIRST REAL RUN: a rate
# near 0 means the buffer never grows and the signal is inert; a rate near 1
# means every frame is stored and max-similarity saturates. Either way the
# threshold, not the mechanism, is what needs moving.
#
# One thing already ruled out: mean-centring the features (correlation instead
# of cosine) changes the gap by 0.001 and is not worth the extra state.
NOVELTY_ADMIT_SIM = float(os.environ.get("NOVELTY_ADMIT_SIM", 0.9))
NOVELTY_BUFFER_MAX = int(os.environ.get("NOVELTY_BUFFER_MAX", 2048))
# v1's visited set resets every episode; this matches it by default.
NOVELTY_PERSIST = os.environ.get("NOVELTY_PERSIST", "0") == "1"


def l2_normalise(v, eps=1e-8):
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), eps)


class ImageNoveltyBuffer:
    """Episodic memory of distinct views, and the novelty of the current one."""

    def __init__(self, admit_sim=NOVELTY_ADMIT_SIM, max_size=NOVELTY_BUFFER_MAX,
                 persist=NOVELTY_PERSIST):
        self.admit_sim = float(admit_sim)
        self.max_size = int(max_size)
        self.persist = bool(persist)
        self._emb = None            # (N, D) L2-normalised, row per stored view
        self.n_admitted = 0
        self.n_queries = 0

    def reset(self):
        """Called at episode start. v1 semantics: forget, unless persisting."""
        if not self.persist:
            self._emb = None
            self.n_admitted = 0
            self.n_queries = 0

    def __len__(self):
        return 0 if self._emb is None else int(self._emb.shape[0])

    def max_similarity(self, embedding):
        """Cosine similarity to the most similar stored view, in [-1, 1].

        An EMPTY buffer returns -1.0 -- maximally novel. Returning 0.0 would
        say "orthogonal to everything", which is a different and weaker claim,
        and would make the first frame of an episode look half-familiar.
        """
        if self._emb is None or self._emb.shape[0] == 0:
            return -1.0
        q = l2_normalise(embedding)
        return float(np.max(self._emb @ q))

    def novelty(self, embedding):
        """1 - max similarity, in [0, 2]. Higher is more novel."""
        return 1.0 - self.max_similarity(embedding)

    def observe(self, embedding):
        """Score this view, admit it if distinct, and return (novelty, admitted).

        Scoring happens BEFORE admission, or every frame would be compared
        against itself and score zero novelty.
        """
        self.n_queries += 1
        sim = self.max_similarity(embedding)
        admitted = sim < self.admit_sim
        if admitted:
            self._admit(embedding)
        return 1.0 - sim, admitted

    def _admit(self, embedding):
        q = l2_normalise(embedding)[None]
        if self._emb is None:
            self._emb = q
        else:
            self._emb = np.concatenate([self._emb, q], axis=0)
            if self._emb.shape[0] > self.max_size:
                # Drop the OLDEST. A random or nearest eviction would quietly
                # change what "novel" means as the buffer fills; FIFO keeps the
                # memory a sliding window over the run rather than a biased
                # sample of it.
                self._emb = self._emb[-self.max_size:]
        self.n_admitted += 1

    def stats(self):
        return {"buffer": len(self),
                "admitted": self.n_admitted,
                "queries": self.n_queries,
                "admit_rate": self.n_admitted / max(1, self.n_queries)}
