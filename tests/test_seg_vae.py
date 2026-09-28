"""ResNet-9 masked segmentation VAE (user 2026-09-28)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from vae_nav import seg_vae as sv  # noqa: E402


def test_shapes_and_encode():
    m = sv.SegVAE(latent_dim=32)
    x = jnp.zeros((2, 120, 120, 3))
    params = m.init(jax.random.PRNGKey(0), x, jax.random.PRNGKey(1))
    logits, mu, logvar = m.apply(params, x, jax.random.PRNGKey(2))
    assert logits.shape == (2, 120, 120, sv.N_CLASSES) and mu.shape == (2, 32)
    z = m.apply(params, x, method=sv.SegVAE.encode)
    assert z.shape == (2, 32)


def test_patch_mask_ratio():
    x = jnp.ones((64, 120, 120, 3))
    y = sv.patch_mask(jax.random.PRNGKey(0), x, ratio=0.75)
    assert abs(float(1 - y.mean()) - 0.75) < 0.03
    cell = np.asarray(y[0, :8, :8, 0])            # a cell is all kept or all masked
    assert cell.min() == cell.max()


def test_soft_targets_keep_thin_lines():
    lab = np.zeros((1, 240, 240), np.uint8)
    lab[0, :, 100] = 2                            # a one-pixel lane line at 240 px
    t = sv.soft_targets(jnp.asarray(lab))
    assert t.shape == (1, 120, 120, sv.N_CLASSES)
    assert np.allclose(np.asarray(t.sum(-1)), 1.0)
    assert float(t[0, :, 50, 2].mean()) == pytest.approx(0.5)


def test_augment_keeps_valid_targets():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, sv.N_CLASSES, (8, 240, 240)).astype(np.uint8)
    t = sv.soft_targets(jnp.asarray(lab))
    x = jnp.asarray(rng.uniform(0, 1, (8, 120, 120, 3)).astype(np.float32))
    xa, ta = sv.augment(jax.random.PRNGKey(0), x, t)
    assert xa.shape == x.shape and ta.shape == t.shape
    assert float(xa.min()) >= 0.0 and float(xa.max()) <= 1.0
    assert np.allclose(np.asarray(ta.sum(-1)), 1.0, atol=1e-4)


def test_loss_decreases_on_one_batch():
    import optax
    m = sv.SegVAE(latent_dim=16)
    rng = np.random.default_rng(0)
    lab = np.zeros((4, 240, 240), np.uint8); lab[:, 120:] = 1; lab[:, :, 110:130] = 2
    t = sv.soft_targets(jnp.asarray(lab))
    x = jnp.asarray(rng.uniform(0, 1, (4, 120, 120, 3)).astype(np.float32))
    p = m.init(jax.random.PRNGKey(0), x, jax.random.PRNGKey(1))
    opt = optax.adam(1e-3); st = opt.init(p)
    w = jnp.ones(sv.N_CLASSES)

    @jax.jit
    def step(p, st, k):
        def lf(p):
            lg, mu, lv = m.apply(p, x, k)
            return sv.seg_vae_loss(lg, t, mu, lv, w, 1e-3)[0]
        l, g = jax.value_and_grad(lf)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    losses = []
    for i in range(30):
        p, st, l = step(p, st, jax.random.PRNGKey(i))
        losses.append(float(l))
    assert losses[-1] < 0.5 * losses[0]
