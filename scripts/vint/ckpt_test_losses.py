#!/usr/bin/env python3
"""Epoch-average test loss of every ViNT checkpoint in a training run folder.

ViNT does not print it (evaluate's last _log_data only prints when the final
batch index is a multiple of print_log_freq), but train_eval_loop stores
avg_total_test_loss in each <epoch>.pth. Loading a checkpoint unpickles the
model, so vint_train must be importable (PYTHONPATH as in vint_train.sbatch).

usage: ckpt_test_losses.py <run dir with 0.pth, 1.pth, ...>
"""
import glob
import os
import re
import sys

import torch


def main():
    run = sys.argv[1]
    rows = []
    for p in glob.glob(os.path.join(run, "*.pth")):
        m = re.fullmatch(r"(\d+)\.pth", os.path.basename(p))
        if not m:
            continue                              # latest.pth duplicates the last epoch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        rows.append((int(m.group(1)), float(ck["avg_total_test_loss"])))
    rows.sort()
    for epoch, loss in rows:
        print(f"epoch {epoch} test_loss {loss:.4f}", flush=True)
    if rows:
        best = min(rows, key=lambda r: r[1])
        print(f"best epoch {best[0]} (test loss {best[1]:.4f}) of {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
