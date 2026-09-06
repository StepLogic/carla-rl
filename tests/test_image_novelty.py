"""Appearance novelty against a buffer of stored views.

Complements the geometric visit ring: the ring needs true pose (privileged),
this needs only the frame. Both are kept so an ablation can say which carries
the behaviour.
"""
import numpy as np
import pytest

from vae_nav.image_novelty import ImageNoveltyBuffer, l2_normalise


def emb(*vals, dim=8):
    """A deterministic embedding pointing in a chosen direction."""
    v = np.zeros(dim, np.float32)
    for i, x in enumerate(vals):
        v[i] = x
    return v


def test_l2_normalise_gives_unit_vectors():
    v = l2_normalise(emb(3.0, 4.0))
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_l2_normalise_survives_a_zero_vector():
    """A dead frame must not divide by zero."""
    v = l2_normalise(np.zeros(8, np.float32))
    assert np.all(np.isfinite(v))


def test_empty_buffer_is_maximally_novel():
    """-1.0, not 0.0.

    Zero would mean 'orthogonal to everything stored', which is a weaker claim
    and would make the first frame of an episode look half-familiar.
    """
    b = ImageNoveltyBuffer()
    assert b.max_similarity(emb(1.0)) == -1.0
    assert b.novelty(emb(1.0)) == 2.0


def test_identical_view_is_maximally_familiar():
    b = ImageNoveltyBuffer()
    b.observe(emb(1.0))
    assert b.max_similarity(emb(1.0)) == pytest.approx(1.0)
    assert b.novelty(emb(1.0)) == pytest.approx(0.0)


def test_orthogonal_view_is_novel():
    b = ImageNoveltyBuffer()
    b.observe(emb(1.0, 0.0))
    assert b.max_similarity(emb(0.0, 1.0)) == pytest.approx(0.0)
    assert b.novelty(emb(0.0, 1.0)) == pytest.approx(1.0)


def test_scoring_happens_before_admission():
    """Otherwise every frame is compared against itself and scores zero."""
    b = ImageNoveltyBuffer()
    novelty, admitted = b.observe(emb(1.0))
    assert admitted is True
    assert novelty == 2.0, "the first frame must score as fully novel"


def test_similarity_ignores_magnitude():
    """Cosine, not dot product.

    Un-normalised conv features track brightness more than content, so a bright
    wall and a bright road would read as similar by magnitude alone.
    """
    b = ImageNoveltyBuffer()
    b.observe(emb(1.0, 0.0))
    assert b.max_similarity(emb(100.0, 0.0)) == pytest.approx(1.0)


def test_near_duplicate_frames_are_not_admitted():
    """THE REASON ADMISSION IS THRESHOLDED.

    Consecutive frames 0.283 m apart are nearly identical. Storing all of them
    makes the buffer a near-duplicate pile, max-similarity saturates at ~1.0
    within a few steps, and the signal dies.
    """
    b = ImageNoveltyBuffer(admit_sim=0.9)
    b.observe(emb(1.0, 0.0))
    for _ in range(50):
        _n, admitted = b.observe(emb(1.0, 0.01))   # essentially the same view
        assert admitted is False
    assert len(b) == 1


def test_distinct_views_are_admitted():
    b = ImageNoveltyBuffer(admit_sim=0.9)
    for i in range(6):
        v = np.zeros(8, np.float32)
        v[i] = 1.0                                  # mutually orthogonal
        _n, admitted = b.observe(v)
        assert admitted is True
    assert len(b) == 6


def test_returning_to_a_seen_place_reports_low_novelty():
    """The behaviour the explorer is meant to use."""
    b = ImageNoveltyBuffer(admit_sim=0.9)
    place_a, place_b = emb(1.0, 0.0), emb(0.0, 1.0)
    b.observe(place_a)
    b.observe(place_b)
    assert b.novelty(place_a) == pytest.approx(0.0, abs=1e-6)


def test_buffer_is_capped_and_evicts_oldest_first():
    """FIFO: a random or nearest eviction would change what 'novel' means as
    the buffer fills."""
    b = ImageNoveltyBuffer(admit_sim=0.9, max_size=3)
    first = np.zeros(8, np.float32); first[0] = 1.0
    b.observe(first)
    for i in range(1, 6):
        v = np.zeros(8, np.float32); v[i] = 1.0
        b.observe(v)
    assert len(b) == 3
    assert b.novelty(first) == pytest.approx(1.0), (
        "the oldest view should have been evicted, so it reads novel again")


def test_reset_forgets_by_default():
    """v1's visited set resets every episode; this matches it."""
    b = ImageNoveltyBuffer()
    b.observe(emb(1.0))
    assert len(b) == 1
    b.reset()
    assert len(b) == 0
    assert b.novelty(emb(1.0)) == 2.0


def test_persist_keeps_the_buffer_across_episodes():
    b = ImageNoveltyBuffer(persist=True)
    b.observe(emb(1.0))
    b.reset()
    assert len(b) == 1


def test_stats_report_admit_rate():
    b = ImageNoveltyBuffer(admit_sim=0.9)
    b.observe(emb(1.0, 0.0))
    for _ in range(9):
        b.observe(emb(1.0, 0.01))
    s = b.stats()
    assert s["queries"] == 10 and s["admitted"] == 1
    assert s["admit_rate"] == pytest.approx(0.1)


def test_discriminates_on_real_mobilenet_embeddings():
    """THE TEST THAT DECIDES WHETHER ANY OF THIS IS USEFUL.

    A novelty signal is inert unless same-place and different-place embeddings
    separate by more than the admission threshold sits from either.

    MEASURED on structured, image-like inputs: same place 0.966, different
    place 0.691, gap 0.275 -- so a 0.9 threshold rejects near-duplicates and
    admits genuinely new views.

    Note on stimulus: uniform random-noise images do NOT work as a test here.
    Every noise image has near-identical conv statistics, giving 0.9999 vs
    0.9799 and a useless 0.02 range. That is a property of the stimulus, not
    of the embedding, and it briefly looked like a real defect.
    """
    src = pytest.importorskip("vae_nav.mobilenet_source")
    enc = src.MobileNetSource(image_size=120, pool=4)

    def scene(seed, shift=0):
        r = np.random.default_rng(seed)
        x = np.linspace(0, 1, 120)[None, :].repeat(120, 0)
        y = np.linspace(0, 1, 120)[:, None].repeat(120, 1)
        base = np.stack([x, y, (x + y) / 2], -1) * 255
        for _ in range(6):
            cx, cy = r.integers(10, 110, 2)
            base[max(0, cy - 20):cy + 20, max(0, cx - 20):cx + 20] = \
                r.integers(0, 256, 3)
        return np.clip(np.roll(base, shift, axis=1), 0, 255).astype(np.uint8)

    b = ImageNoveltyBuffer(admit_sim=0.9)
    b.observe(enc.latent(scene(0)))

    same = b.max_similarity(enc.latent(scene(0, shift=2)))
    diff = b.max_similarity(enc.latent(scene(1)))

    assert same > 0.9, f"a near-duplicate view scored {same:.4f}; it would be admitted"
    assert diff < 0.9, f"a different view scored {diff:.4f}; it would be rejected"
    assert same - diff > 0.1, (
        f"gap {same - diff:.4f} is too small to threshold reliably")


def test_works_at_the_mobilenet_embedding_width():
    """384-d is what the MobileNetV3 backend actually emits."""
    rng = np.random.default_rng(0)
    b = ImageNoveltyBuffer(admit_sim=0.9)
    for _ in range(20):
        b.observe(rng.normal(size=384).astype(np.float32))
    assert len(b) > 1
    assert -1.0 <= b.max_similarity(rng.normal(size=384).astype(np.float32)) <= 1.0
