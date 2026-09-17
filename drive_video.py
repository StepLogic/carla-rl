"""Two-panel driving video for drive_plan_c.py: policy camera | top-down plan.

No VAE decode (a per-frame GPU decode killed CARLA mid-sweep before); only cv2
drawing on a precomputed map background, so recording does not change what the
car does or how fast the simulator runs in any way that matters.
"""
import numpy as np

PANEL = 512
FPS = 20


class TopDown:
    """World (x, y) -> panel pixels, with a cached background of lanes, edges and nodes."""

    def __init__(self, lane_lines, node_xy, edges, pad_m=15.0, size=PANEL):
        import cv2
        pts = np.concatenate([np.asarray(l, np.float64)[:, :2] for l in lane_lines if len(l)])
        self.lo = pts.min(0) - pad_m
        span = (pts.max(0) + pad_m) - self.lo
        self.scale = (size - 1) / float(span.max())
        self.size = size
        self.node_xy = np.asarray(node_xy, np.float64)
        bg = np.full((size, size, 3), 255, np.uint8)
        for l in lane_lines:
            if len(l) > 1:
                cv2.polylines(bg, [self.px(np.asarray(l)[:, :2])], False, (214, 214, 214), 3, cv2.LINE_AA)
        for u, v in edges:
            cv2.line(bg, tuple(self.px(self.node_xy[u])), tuple(self.px(self.node_xy[v])), (170, 170, 170), 1, cv2.LINE_AA)
        for p in self.px(self.node_xy):
            cv2.circle(bg, tuple(p), 3, (110, 110, 110), -1, cv2.LINE_AA)
        self.bg = bg

    def px(self, xy):
        """World metres -> integer pixels. CARLA y grows 'down' on the top-down plots, so no flip."""
        xy = np.asarray(xy, np.float64)
        p = np.round((xy - self.lo) * self.scale).astype(np.int32)
        return np.clip(p, 0, self.size - 1)


def render_frame(td, rgb, trail, path, i, target, goal_xy, start_node, goal_node, text_lines):
    """-> (PANEL, 2*PANEL, 3) BGR uint8 frame."""
    import cv2
    left = np.zeros((PANEL, PANEL, 3), np.uint8)
    if rgb is not None:
        a = np.asarray(rgb)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        left = cv2.resize(a[..., :3][..., ::-1], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    right = td.bg.copy()
    if len(path) > 1:
        cv2.polylines(right, [td.px(td.node_xy[path])], False, (60, 170, 60), 3, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(right, [td.px(np.asarray(trail))], False, (200, 120, 30), 2, cv2.LINE_AA)
    cv2.circle(right, tuple(td.px(td.node_xy[start_node])), 7, (40, 40, 220), 2, cv2.LINE_AA)
    cv2.circle(right, tuple(td.px(td.node_xy[goal_node])), 9, (40, 40, 220), 2, cv2.LINE_AA)
    cv2.drawMarker(right, tuple(td.px(goal_xy)), (0, 0, 0), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
    cv2.drawMarker(right, tuple(td.px(target)), (0, 190, 230), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
    if len(trail):
        cv2.circle(right, tuple(td.px(trail[-1])), 6, (0, 0, 230), -1, cv2.LINE_AA)
    frame = np.concatenate([left, right], axis=1)
    for k, t in enumerate(text_lines):
        y = 20 + 20 * k
        cv2.putText(frame, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    legend = ["Plan", "Driven", "Target", "Goal", "Localised start/goal"]
    cols = [(60, 170, 60), (200, 120, 30), (0, 190, 230), (0, 0, 0), (40, 40, 220)]
    for k, (t, c) in enumerate(zip(legend, cols)):
        y = PANEL - 12 - 18 * (len(legend) - 1 - k)
        cv2.line(frame, (PANEL + 10, y - 4), (PANEL + 30, y - 4), c, 3)
        cv2.putText(frame, t, (PANEL + 36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return frame


def open_writer(path, size=(2 * PANEL, PANEL)):
    import cv2
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    if not w.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    return w
