"""List the ViNT Town07 cells with the fields a table-builder checks: towns, episode towns,
n, SR, checkpoint, episodes file."""
import glob
import json
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
for f in sorted(glob.glob(os.path.join(base, "bench_vintd3_t07_*", "eval_*", "eval.json"))):
    j = json.load(open(f))
    ep_towns = sorted({e.get("town") for e in j["episodes_detail"]})
    print(f"{f.split(os.sep)[-3]:44s} towns={j.get('towns')} episode towns={ep_towns} "
          f"n={j['episodes']} SR={j['success_rate']:.2f} "
          f"ckpt={os.path.basename(j['checkpoint'].rstrip('/'))} "
          f"file={os.path.basename(j.get('episodes_file') or '-')}")
