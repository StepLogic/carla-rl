"""
NoMaD policy adapted for DenseMapNavigator used in CoRL 2026 evaluation.

Duplicate of navigation_policies/baseline_policies/nomad_policy.py, modified to:
  - Accept a DenseMapNavigator + path instead of a map_dir of numbered jpgs
  - Load goal images directly from the navigator (no disk I/O)
  - Support external subgoal index updates from the evaluation loop
"""

from collections import deque
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from navigation_policies.baseline_policies.misc import (
    get_action, Args, to_numpy, transform_images, load_model
)
from navigation_policies.baseline_policies.baselines_config import nomad_config
from PIL import Image

EPS = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args(model="nomad", waypoint=3, num_samples=8, radius=3, close_threshold=3)


def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi


def pd_controller(waypoint: np.ndarray):
    """PD controller for the robot"""
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx))
    elif np.abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2)
    else:
        v = dx
        w = np.arctan(dy / dx)
    v = np.clip(v, 0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    return w, v


class NoMaD_DenseMap:
    """NoMaD policy that works with DenseMapNavigator instead of a map_dir on disk.

    Instead of loading numbered jpg files from a directory, this class takes a
    DenseMapNavigator and a path (list of node indices). Goal images are fetched
    from the navigator's get_subgoal_image() method, avoiding intermediate disk I/O.

    The external evaluation loop can advance the subgoal index via set_subgoal_idx().
    """

    def __init__(self, ckpt_path=None, nav=None, path=None,
                 search_radius=3, threshold=None, traj_id=None):
        self.closest_node = 0
        self.traj_id = traj_id
        self.path = path or []
        self.utilized_nodes = []
        self.filtered_nodes = []
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
        self._subgoal_idx = 1  # externally controlled, start toward 2nd node

        # Load model parameters
        self.context_size = nomad_config.get("context_size", 3)
        self.model_params = nomad_config
        self.context_queue = []
        self.possible_goals = deque(maxlen=self.context_size + 1)

        # Load model weights
        self.model = load_model(ckpt_path, nomad_config, device)
        self.model = self.model.to(device)
        self.model.eval()

        self.num_diffusion_iters = nomad_config["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=nomad_config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # Pre-load topomap images from navigator
        self.topomap = []
        for node_idx in self.nav_path:
            img = self.nav.get_subgoal_image(node_idx)
            if img is not None:
                self.topomap.append(Image.fromarray(img))
            else:
                # Fallback: black image
                self.topomap.append(Image.new("RGB", (96, 96), (0, 0, 0)))

        self.goal_node = len(self.topomap) - 1
        self._last_goal_image = None  # cache last goal image for reuse
        self.dists = []
        self.actions = None

        print(f"TOPOMAP SIZE (from navigator): {len(self.topomap)}")
        print("===============.....Model Loaded .... Waiting for Observations.....==============================")

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
                self.topomap.append(Image.new("RGB", (96, 96), (0, 0, 0)))
        self.goal_node = len(self.topomap) - 1

    def callback_obs(self, msg):
        """Process incoming observation"""
        obs_img = Image.fromarray(msg[..., -1].astype(np.uint8))

        if self.context_size is not None:
            self.context_queue.append(obs_img)

        if len(self.context_queue) > self.context_size + 1:
            self.possible_goals.append(self.context_queue.pop(0))

    def eval_action(self, obs, goal_image=None):
        """Evaluate action given current observation and optional goal image.

        Args:
            obs: BGR observation array with trailing dimension for stacking.
            goal_image: Optional RGB numpy array from the dense mapper path.
                        When provided, it is cached and used directly as the
                        goal. When None, the previously cached goal image is
                        reused instead of falling back to the search window.

        Returns:
            Waypoint action array.
        """
        self.callback_obs(obs)

        if len(self.context_queue) > self.model_params["context_size"]:
            # if abs(self.closest_node - self.goal_node) < 2:
            #     self.closest_node = self.goal_node
            #     return np.zeros((4, 2))

            if goal_image is not None:
                self._last_goal_image = goal_image

            # Transform observation images
            obs_images = transform_images(
                list(self.context_queue), self.model_params["image_size"], center_crop=False
            )
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1)
            obs_images = obs_images.to(device)
            mask = torch.zeros(1).long().to(device)

            if self._last_goal_image is not None:
                # Use goal image directly — from this call or cached from prior call
                goal_pil = Image.fromarray(self._last_goal_image) if not isinstance(self._last_goal_image, Image.Image) else self._last_goal_image
                goal_img_tensor = transform_images(
                    goal_pil, self.model_params["image_size"], center_crop=False
                ).to(device)

                # Encode observation and single goal
                obsgoal_cond = self.model(
                    'vision_encoder',
                    obs_img=obs_images.repeat(1, 1, 1, 1),
                    goal_img=goal_img_tensor,
                    input_goal_mask=mask.repeat(1)
                )

                # Infer action using diffusion
                with torch.no_grad():
                    if len(obsgoal_cond.shape) == 2:
                        obsgoal_cond = obsgoal_cond.repeat(self.args.num_samples, 1)
                    else:
                        obsgoal_cond = obsgoal_cond.repeat(self.args.num_samples, 1, 1)

                    noisy_action = torch.randn(
                        (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                        device=device
                    )
                    naction = noisy_action

                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    for k in self.noise_scheduler.timesteps[:]:
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obsgoal_cond
                        )
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                naction = to_numpy(get_action(naction))
                return naction[1]

            # Fallback: search window around the externally-set subgoal
            center = min(self._subgoal_idx, self.goal_node)
            start = max(center - self.args.radius, 0)
            end = min(center + self.args.radius + 1, self.goal_node + 1)

            # Transform goal images
            goal_image = [
                transform_images(g_img, self.model_params["image_size"], center_crop=False).to(device)
                for g_img in self.topomap[start:end]
            ]
            goal_image = torch.concat(goal_image, dim=0)

            # Encode observation and goals
            obsgoal_cond = self.model(
                'vision_encoder',
                obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                goal_img=goal_image,
                input_goal_mask=mask.repeat(len(goal_image))
            )

            # Predict distances
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            self.dists = dists

            # Find closest node in search window
            min_idx = np.argmin(dists)
            self.closest_node = min_idx + start

            # Select subgoal based on distance threshold
            if dists[min_idx] < 3.0:
                sg_idx = min_idx
                self.filtered_nodes.append(self.closest_node)
            else:
                self.filtered_nodes.append(
                    start + min(min_idx + 1, len(obsgoal_cond) - 1)
                )
                sg_idx = min(min_idx + 1, len(obsgoal_cond) - 1)

            obsgoal_cond = obsgoal_cond[sg_idx].unsqueeze(0)

            # Infer action using diffusion
            with torch.no_grad():
                if len(obsgoal_cond.shape) == 2:
                    obsgoal_cond = obsgoal_cond.repeat(self.args.num_samples, 1)
                else:
                    obsgoal_cond = obsgoal_cond.repeat(self.args.num_samples, 1, 1)

                noisy_action = torch.randn(
                    (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                    device=device
                )
                naction = noisy_action

                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                for k in self.noise_scheduler.timesteps[:]:
                    noise_pred = self.model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obsgoal_cond
                    )
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            naction = to_numpy(get_action(naction))
            # print(naction)
            return naction[1]
        else:
            return np.zeros((4, 2))