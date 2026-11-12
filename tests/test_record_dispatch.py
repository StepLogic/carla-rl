"""Expert recording must not trip the junction-ramp guard (imitation-first plan,
Task 5). v30's recipe sets JUNCTION_RAMP_START, which main() refused outside the
parallel training loop -- correctly for training, wrongly for recording, which
like eval keeps the overlay's fixed junction fraction. The first cluster smoke
died on exactly this (job 463137)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from train_carla_sac import ramp_needs_parallel  # noqa: E402


def test_recording_is_exempt_from_the_junction_ramp_guard():
    assert not ramp_needs_parallel(0.5, parallel=False, eval_from="", expert_record=True)


def test_serial_training_with_a_ramp_is_still_refused():
    assert ramp_needs_parallel(0.5, parallel=False, eval_from="", expert_record=False)


def test_eval_stays_exempt():
    assert not ramp_needs_parallel(0.5, parallel=False, eval_from="ckpt", expert_record=False)


def test_parallel_training_or_no_ramp_is_fine():
    assert not ramp_needs_parallel(0.5, parallel=True, eval_from="", expert_record=False)
    assert not ramp_needs_parallel(None, parallel=False, eval_from="", expert_record=False)


def test_bc_train_dispatches_before_any_server_or_encoder(tmp_path, monkeypatch):
    """BC_TRAIN=1 (Task 6) clones from shards: it must hand run_bc_train the
    trainer's own build_agent and return before CARLA or DEBiT is touched."""
    import torch_rl.bc_trainer as bct
    import torch_rl.carla_server as srv
    import torch_rl.debit_encoder as enc
    import train_carla_sac as T

    calls = []
    monkeypatch.setattr(bct, "run_bc_train", lambda out, make_agent=None: calls.append(
        (out, make_agent)))

    def boom(*a, **k):
        raise AssertionError("BC_TRAIN started a simulator or encoder")
    monkeypatch.setattr(srv, "CarlaServer", boom)
    monkeypatch.setattr(enc, "DebitTinyEncoder", boom)
    monkeypatch.setattr(T, "DEVICE", "cpu")
    monkeypatch.setattr(T, "OUT", str(tmp_path / "bc_run"))
    monkeypatch.setenv("BC_TRAIN", "1")
    T.main()
    assert calls == [(str(tmp_path / "bc_run"), T.build_agent)]
