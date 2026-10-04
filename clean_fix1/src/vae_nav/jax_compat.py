"""One-line JAX compatibility alias, importable so it is not copy-pasted.

distrax 0.1.5 against jax 0.6.2: distrax's jacobian inference reaches for
`jax.core.Var`, which JAX removed in 0.6.0. Only the rescaled-tanh policy path
touches it -- which is exactly the path taken when `action_space` is passed, and
that is the path every arm in this repo needs, because our action space is
asymmetric (throttle has no reverse) and an unbounded tanh head would put half
its support out of reach.

Aliasing the symbol is enough: the class moved, it did not change.

MUST BE IMPORTED BEFORE THE POLICY IS CONSTRUCTED. train_carla_tqc.py carries
the same three lines inline near the top of the file; this module exists so a
second consumer -- the gated learner, and any test that builds an agent -- does
not have to copy them and then drift from the original.

Import for the side effect:

    import vae_nav.jax_compat  # noqa: F401
"""
import jax
import jax.extend.core

if not hasattr(jax.core, "Var"):
    jax.core.Var = jax.extend.core.Var
