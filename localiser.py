"""Where the features for node matching come from (user 2026-09-19: "replace localization with dinov2").

plan_c.nearest_latent picks the node whose stored feature is nearest the live one by cosine. What
changes between these three is only WHICH feature vector each side supplies:

  mobilenet_trace  the shipped behaviour -- nodes are the trace's stored latents (frozen ImageNet
                   MobileNet-V3-Small, 384-d), recorded while the car was DRIVING past them; the query
                   is the live latent, and the goal is the stored episode-view latent.
  mobilenet_bank   the same encoder, but nodes come from the teleported RGB bank (settled car, trace
                   yaw). Isolates the capture change from the encoder change.
  dinov2_bank      DINOv2 ViT-S/14 over the same bank frames, and the live camera frame embedded on
                   the fly. CPU is fine: matching happens at the start, the goal and re-localisations,
                   not every step.

Any comparison of the last two against the first must keep in mind that the first differs in BOTH the
encoder and the capture, which is exactly why mobilenet_bank exists.
"""
import numpy as np

MODES = ("mobilenet_trace", "mobilenet_bank", "dinov2_bank")
HUB = "/home/robotlab/.cache/torch/hub/facebookresearch_dinov2_main"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Localiser:
    def __init__(self, mode, trace, views, feats=None, pooling="patchmean", model="dinov2_vits14", size=224):
        if mode not in MODES:
            raise SystemExit(f"--localiser must be one of {MODES}")
        self.mode, self.pooling, self.size = mode, pooling, size
        self.views = views
        if mode == "mobilenet_trace":
            self.F = np.stack([np.asarray(o["latent"], np.float64) for o in trace["observations"]])
            self.row = {i: i for i in range(len(self.F))}
            self.vrow = None
            return
        z = np.load(feats, allow_pickle=True)
        kind, idx, role = z["kind"], z["index"], z["role"]
        self.F = np.asarray(z["latent" if mode == "mobilenet_bank" else pooling], np.float64)
        self.row = {int(i): k for k, (kd, i) in enumerate(zip(kind, idx)) if kd == "trace"}
        self.vrow = {(str(kd), int(i), str(r)): k for k, (kd, i, r) in enumerate(zip(kind, idx, role))
                     if kd != "trace"}
        self.model = None
        if mode == "dinov2_bank":
            import torch
            self.torch = torch
            self.model = torch.hub.load(HUB, model, source="local", pretrained=True).eval()
            print(f"[localiser] dinov2 {model} on cpu, pooling={pooling}, bank={feats}", flush=True)

    def nodes(self, tidx):
        """Feature matrix for the map's nodes, in node order."""
        missing = [int(t) for t in tidx if int(t) not in self.row]
        if missing:
            raise SystemExit(f"{len(missing)} node trace indices are not in the bank (e.g. {missing[:5]})")
        return self.F[[self.row[int(t)] for t in tidx]]

    def _embed(self, rgb):
        t = self.torch
        x = t.from_numpy(np.asarray(rgb, np.float32)[None] / 255.0).permute(0, 3, 1, 2)
        x = t.nn.functional.interpolate(x, size=(self.size, self.size), mode="bicubic",
                                        align_corners=False).clamp(0, 1)
        x = (x - t.tensor(MEAN).view(1, 3, 1, 1)) / t.tensor(STD).view(1, 3, 1, 1)
        with t.no_grad():
            o = self.model.forward_features(x)
        v = o["x_norm_clstoken"] if self.pooling == "cls" else o["x_norm_patchtokens"].mean(1)
        return v[0].numpy().astype(np.float64)

    def query(self, obs, info):
        """The live feature to match against the nodes."""
        if self.mode == "dinov2_bank":
            rgb = info.get("rgb")
            if rgb is None:
                raise SystemExit("info['rgb'] is missing -- dinov2 localisation needs the camera frame")
            return self._embed(rgb)
        return np.asarray(obs["latent"], np.float64)

    def goal(self, set_name, index):
        """The captured goal view's feature."""
        if self.mode == "mobilenet_trace":
            return np.asarray(self.views[(set_name, index, "goal")]["obs"]["latent"], np.float64)
        k = self.vrow.get((set_name, int(index), "goal"))
        if k is None:
            raise SystemExit(f"goal view {set_name}#{index} is not in the bank")
        return self.F[k]
