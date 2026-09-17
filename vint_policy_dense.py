"""
ViNT policy adapted for DenseMapNavigator used in CoRL 2026 evaluation.

Duplicate of navigation_policies/baseline_policies/vint_policy.py, modified to:
  - Accept a DenseMapNavigator + path instead of a map_dir of numbered jpgs
  - Load goal images directly from the navigator (no disk I/O)
  - Support external subgoal index updates from the evaluation loop
"""

from collections import deque
import os
import numpy as np
import torch
from navigation_policies.baseline_policies.misc import (
    Args, to_numpy, transform_images, load_model
)
from navigation_policies.baseline_policies.baselines_config import vint_config
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args(model="vint", waypoint=3, num_samples=8, radius=3, close_threshold=3)


class ViNT_Policy_DenseMap:
    """ViNT policy that works with DenseMapNavigator instead of a map_dir on disk.

    Instead of loading numbered jpg files from a directory, this class takes a
    DenseMapNavigator and a path (list of node indices). Goal images are fetched
    from the navigator's get_subgoal_image() method, avoiding intermediate disk I/O.

    The external evaluation loop can advance the subgoal index via set_subgoal_idx().
    """

    def __init__(self, ckpt_path=None, nav=None, path=None,
                 model_name="vint", search_radius=4, threshold=1.0, traj_id=None):
        self.closest_node = 0
        self.args = args
        self.args = self.args._replace(radius=search_radius)
        if threshold is not None:
            self.args = self.args._replace(close_threshold=threshold)

        if ckpt_path is None:
            raise FileNotFoundError("Please specify checkpoint path")
        if nav is None:
            raise ValueError("Please provide a DenseMapNavigator instance")

        self.nav = nav
        self.nav_path = path if path is not None else []
        self._subgoal_idx = 1
        self.traj_id = traj_id

        self.context_size = vint_config["context_size"]
        self.model_params = vint_config
        self.context_queue = deque(maxlen=self.context_size + 1)
        vint_config["model_type"] = model_name

        self.utilized_nodes = []
        self.filtered_nodes = []

        # Load model weights
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
        self.model = load_model(ckpt_path, vint_config, device)
        self.model = self.model.to(device)
        self.model.eval()

        # Pre-load topomap images from navigator
        self.topomap = []
        for node_idx in self.nav_path:
            img = self.nav.get_subgoal_image(node_idx)
            if img is not None:
                self.topomap.append(Image.fromarray(img))
            else:
                self.topomap.append(Image.new("RGB", (85, 64), (0, 0, 0)))

        self.goal_node = len(self.topomap) - 1
        self.closest_node = 0
        self._last_goal_image = None  # cache last goal image for reuse
        print("===================== .... Model Loaded .... Waiting for Observations .... ==============================")

    def set_subgoal_idx(self, idx):
        """Update the subgoal index from the external evaluation loop."""
        self._subgoal_idx = idx

    def update_path(self, path):
        """Replace the navigation path and reload topomap images."""
        self.nav_path = path
        self.topomap = []
        for node_idx in self.nav_path:
            img = self.nav.get_subgoal_image(node_idx)
            if img is not None:
                self.topomap.append(Image.fromarray(img))
            else:
                self.topomap.append(Image.new("RGB", (85, 64), (0, 0, 0)))
        self.goal_node = len(self.topomap) - 1

    def callback_obs(self, msg):
        obs_img = Image.fromarray(msg[..., -1].astype(np.uint8))
        if self.context_size is not None:
            self.context_queue.append(obs_img)

    def eval_action(self, obs, goal_image=None):
        """Evaluate action given observation and optional goal image.

        Args:
            obs: BGR observation array with trailing dimension for stacking.
            goal_image: Optional RGB numpy array from the dense mapper path.
                        When provided, it is cached and used directly as the
                        goal. When None, the previously cached goal image is
                        reused instead of falling back to the search window.

        Returns:
            Waypoint action array of shape (4, 2).
        """
        self.callback_obs(obs)
        if len(self.context_queue) > self.model_params["context_size"]:
            # if abs(self.closest_node - self.goal_node) < 2:
            #     self.closest_node = self.goal_node
            #     return np.zeros((4, 2))

            if goal_image is not None:
                self._last_goal_image = goal_image

            if self._last_goal_image is not None:
                # Use goal image directly — from this call or cached from prior call
                transf_obs_img = transform_images(
                    list(self.context_queue), self.model_params["image_size"]
                )
                goal_pil = Image.fromarray(self._last_goal_image) if not isinstance(self._last_goal_image, Image.Image) else self._last_goal_image
                goal_data = transform_images(goal_pil, self.model_params["image_size"])

                transf_obs_img = transf_obs_img.to(device)
                goal_data = goal_data.to(device)

                distances, waypoints = self.model(transf_obs_img, goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)

                wp = np.zeros((4, 2))
                wp[:4, :] = waypoints[0, :4, :2]
                return wp
            else:
                # Search window centred on external subgoal index
                center = min(self._subgoal_idx, self.goal_node)
                start = max(center - self.args.radius, 0)
                end = min(center + self.args.radius + 1, self.goal_node + 1)

                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(self.topomap[start:end]):
                    transf_obs_img = transform_images(
                        list(self.context_queue), self.model_params["image_size"]
                    )
                    goal_data = transform_images(sg_img, self.model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)

                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)

                mind_idx = np.argmin(distances)
                self.closest_node = min(start + mind_idx, len(self.topomap) - 1)
                self.filtered_nodes.append(self.closest_node)

                if distances[mind_idx] > 3.0:
                    waypoints = waypoints[mind_idx]
                else:
                    waypoints = waypoints[min(mind_idx + 1, len(waypoints) - 1)]

                wp = np.zeros((4, 2))
                wp[:4, :] = waypoints[:4, :2]
                return wp
        else:
            return np.zeros((4, 2))