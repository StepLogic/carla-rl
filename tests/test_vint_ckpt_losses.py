"""ViNT prints no epoch-average test loss (its last _log_data call only prints
when the final batch index is a multiple of print_log_freq), but every epoch
checkpoint stores avg_total_test_loss: ckpt_test_losses.py reads them back."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch = pytest.importorskip("torch")


def test_it_lists_every_epoch_with_its_test_loss_in_epoch_order(tmp_path):
    for epoch, loss in ((0, 2.5), (1, 1.75), (10, 1.5)):
        torch.save({"avg_total_test_loss": loss, "model": None}, tmp_path / f"{epoch}.pth")
    torch.save({"avg_total_test_loss": 9.0}, tmp_path / "latest.pth")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "vint", "ckpt_test_losses.py"),
                        str(tmp_path)], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr
    rows = [ln.split() for ln in r.stdout.strip().splitlines() if ln.startswith("epoch")]
    assert [(int(a[1]), float(a[3])) for a in rows] == [(0, 2.5), (1, 1.75), (10, 1.5)]
    assert "best epoch 10" in r.stdout
