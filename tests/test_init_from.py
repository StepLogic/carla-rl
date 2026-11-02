"""INIT_FROM: resume training from a checkpoint (user 2026-11-01, "resume v30
and train for longer").

A checkpoint restores every module and target, both temperatures, all four
optimisers and the update counter -- everything except the REPLAY BUFFER, which
is not saved. Updating straight away on a near-empty buffer is what destroyed
goal steering in an earlier warm start (corr 0.57 -> 0.04 by 50k), so the
trainer refuses INIT_FROM unless LEARNING_STARTS lets the loaded policy refill
the buffer first.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src():
    return open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_the_checkpoint_loads_after_the_pretrained_init():
    """The checkpoint carries the trained encoder; loading it BEFORE the DEBiT
    initialisation would let the pretrained weights overwrite it."""
    s = _src()
    assert s.index("mono_report = load_debit_mono(agent.compress, TINY)") < \
        s.index("agent.load(INIT_FROM)")


def test_a_short_refill_is_refused():
    s = _src()
    assert 'INIT_FROM = os.environ.get("INIT_FROM", "")' in s
    assert "LEARNING_STARTS < INIT_MIN_LEARNING_STARTS" in s


def test_init_and_eval_are_exclusive():
    assert "INIT_FROM and EVAL_FROM" in _src()
