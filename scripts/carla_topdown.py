#!/usr/bin/env python3
"""Top-down render of a CARLA town, for overlaying maps on the real town.

One RGB camera straight down (pitch -90, yaw 0) at height z above the town's
centre. For a ground point (X, Y, 0) the pixel is exact (pinhole onto a plane):
    f = W / (2 tan(fov/2));  u = W/2 + f (Y - cy) / z;  v = H/2 - f (X - cx) / z
(yaw 0: image up = world +X, image right = world +Y). Buildings lean outward
(perspective); the road surface does not. Writes <out>.png and <out>.json.

    carla_topdown.py --town Town02 --cx 95 --cy 205 --span 240 --out logs/topdown_Town02
"""
import argparse
import json
import math
import os
import queue
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from torch_rl.carla_server import CarlaServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--span", type=float, required=True, help="metres the image must cover")
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--px", type=int, default=2400)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import carla
    z = (a.span / 2.0) / math.tan(math.radians(a.fov) / 2.0)
    srv = CarlaServer(a.port, quality="Epic", log_dir=os.path.dirname(os.path.abspath(a.out)))
    srv.start()
    print(f"[TopDown] CARLA ready in {srv.wait_ready():.0f}s", flush=True)
    try:
        client = carla.Client("localhost", a.port)
        client.set_timeout(120.0)
        world = client.load_world(a.town)
        s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05
        world.apply_settings(s)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(a.px)); bp.set_attribute("image_size_y", str(a.px))
        bp.set_attribute("fov", str(a.fov))
        tf = carla.Transform(carla.Location(x=a.cx, y=a.cy, z=z), carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        cam = world.spawn_actor(bp, tf)
        q = queue.Queue(); cam.listen(q.put)
        img = None
        for _ in range(30):                     # let streaming and lighting settle
            world.tick()
            try:
                img = q.get(timeout=10.0)
            except queue.Empty:
                pass
        arr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1]
        from PIL import Image
        Image.fromarray(arr).save(a.out + ".png")
        json.dump({"town": a.town, "cx": a.cx, "cy": a.cy, "z": z, "fov": a.fov,
                   "W": int(img.width), "H": int(img.height),
                   "projection": "u = W/2 + f*(Y-cy)/z; v = H/2 - f*(X-cx)/z; f = W/(2 tan(fov/2))"},
                  open(a.out + ".json", "w"), indent=1)
        print(f"[TopDown] {a.town}: camera at ({a.cx}, {a.cy}, {z:.1f} m), fov {a.fov} -> {a.out}.png", flush=True)
        cam.stop(); cam.destroy()
        s.synchronous_mode = False; world.apply_settings(s)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
