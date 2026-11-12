"""Which encoder localises better, offline (user 2026-09-19 "replace localization with dinov2").

For every map and every episode start/goal view, each encoder picks the node with the highest cosine
similarity -- exactly what plan_c.nearest_latent does at drive time -- and we score the distance from
that node to the view's TRUE pose. No simulator time is needed to answer this.

The score that matters is EXCESS error: the map's own nearest node is a floor set by how sparse the map
is, so error - floor is the part the encoder is responsible for.
"""
import argparse, json, pickle
import numpy as np

MAPS = ["uniform15_trace", "hybrid_uniform15_entropy_thin4", "sgm_g24_e55_c55", "hybrid_sgm_g44_entropy_thin4",
        "uniform8_trace", "hybrid_uniform8_entropy_thin4", "sgm_g10_e55_c55", "hybrid_sgm_g14_entropy_thin4",
        "uniform6_trace", "hybrid_uniform6_entropy_thin4", "sgm_g7_e55_c55", "hybrid_sgm_g9_entropy_thin4",
        "uniform2_trace", "hybrid_uniform2_entropy_thin4", "sgm_g2_e55_c55", "hybrid_sgm_g2_entropy_thin4"]


def unit(A):
    A = np.asarray(A, np.float64)
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="results/episodes/town02_speed10_s7_dinov2.npz")
    ap.add_argument("--seq-dir", default="results/maps/sequence",
                    help="the graphs the drive actually loads -- every arm has one, driving_map.json does not")
    ap.add_argument("--maps", nargs="*", default=MAPS)
    ap.add_argument("--out", default="tables/localisation_dinov2_town02.md")
    a = ap.parse_args()

    z = np.load(a.feats, allow_pickle=True)
    kind, idx, role, gt = z["kind"], z["index"], z["role"], z["gt_xy"]
    enc = {"MobileNet (current)": z["latent"], "DINOv2 CLS": z["cls"], "DINOv2 patch-mean": z["patchmean"]}
    tr_row = {int(i): k for k, (kd, i) in enumerate(zip(kind, idx)) if kd == "trace"}
    views = [k for k in range(len(kind)) if kind[k] != "trace"]

    md = ["# Localisation error by encoder, Town02 (80 episode start/goal views)\n",
          "Error = distance from the node the encoder picks to the view's true pose. Floor = distance to the "
          "map's nearest node (what perfect matching would give). Hit = share of views localised within 10 m.\n",
          "CAVEAT: node features here come from the teleported RGB bank, not from the trace latents the live "
          "drive matches against, so the absolute numbers are NOT the drive's. Encoders are compared on "
          "identical inputs, which is what this table is for.\n",
          "| Map | Nodes | Floor | " + " | ".join(f"{e} med / p90 / hit" for e in enc) + " |",
          "|---|---|---|" + "---|" * len(enc)]
    for m in a.maps:
        try:
            nodes = json.load(open(f"{a.seq_dir}/{m}_sequence.json"))["nodes"]
        except FileNotFoundError:
            print(f"[skip] {m}: no sequence map"); continue
        ti = [n["trace_index"] for n in nodes]
        if any(t not in tr_row for t in ti):
            print(f"[skip] {m}: {sum(1 for t in ti if t not in tr_row)} nodes missing from the bank")
            continue
        rows = np.array([tr_row[t] for t in ti])
        NXY = np.array([[n["x_m"], n["y_m"]] for n in nodes])
        floor = np.median([np.linalg.norm(NXY - gt[v], axis=1).min() for v in views])
        cells = []
        for name, F in enc.items():
            N = unit(F[rows]); Q = unit(F[views])
            pick = np.argmax(Q @ N.T, axis=1)
            err = np.linalg.norm(NXY[pick] - gt[views], axis=1)
            cells.append(f"{np.median(err):.1f} / {np.percentile(err, 90):.1f} m / {100*np.mean(err < 10):.0f}%")
        md.append(f"| {m} | {len(nodes)} | {floor:.1f} m | " + " | ".join(cells) + " |")
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
