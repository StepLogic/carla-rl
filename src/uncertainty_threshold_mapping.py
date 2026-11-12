# %%
from collections import defaultdict, deque
from functools import partial
import glob
import os
import pickle
import random
import carla
import cv2
import numpy as np
from absl import app, flags
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import carla_location_to_np_array, ndarray_to_location
from carla_eval import CarlaEvalEnv
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy
from agent_wrapper import SetPointAgent
from PIL import Image
from rlib_integration.agent import BasicAgent
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


model_type="vint"
models={
    "gnm":GNM_Policy,
    "nomad":NoMaD,
    "vint":ViNT_Policy
}
default_checkpointts={
    "gnm":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm_modified.pth",
    "nomad":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
    "vint":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth"
}
MODEL=models[model_type]
checkpoint_path=default_checkpointts[model_type]
# https://github.com/carla-simulator/carla/issues/2832

# termination_budget=10
# while termination_budget>0:
done = False
episode_reward = 0
episode_length = 0
goal_location=None
shortest_distance_along_road=1e-8
map_dir="/home/robotlab/scratch/carla-rl/topomap"
# if mode !="dense":
# if map_dir is None:
#     if model_type != "nomad":
#         raise ValueError("Only NoMaD can explore")
#     agent = MODEL(ckpt_path=checkpoint_path,mode="explore")
# else:
#     
#     print(f"Using {len(agent.topomap)} Nodes")
threshold=1.0
# similarity_matrix = "/home/robotlab/scratch/carla-rl/cache/gnm/distance_matrix.pth"
agent = MODEL(ckpt_path=checkpoint_path,mode="navigate",map_dir=map_dir,model_name="vint_modified")
agent.compute_uncertainty_matrix()
# # while True:
# #     pass
# base="/home/robotlab/scratch/carla-rl/topomap"
# results_store="/home/robotlab/scratch/carla-rl/evaluation_v2"

# with open(f"{base}/traj.pkl","rb") as pipe:
#     traj_data=pickle.load(pipe)
# locations = traj_data["locations"]
# print(len(locations))
# for _ in range(4):
#     print(random.choices(list(range(len(locations))),k=2))