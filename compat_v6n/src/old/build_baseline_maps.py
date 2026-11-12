from collections import defaultdict, deque
import glob
import math
import os
import pickle
import random
import cv2
import numpy as np
from absl import app, flags
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import ndarray_to_location
from carla_eval import CarlaEvalEnv
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy
from agent_wrapper import SetPointAgent
from PIL import Image

# task
# tune pid controller
# fix

os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# Define flags
FLAGS = flags.FLAGS
# flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_enum('model', 'nomad', ['nomad', 'gnm','vint'], 'Model type')
flags.DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Whether to use deterministic actions")
flags.DEFINE_string("town", "Town01", "Town Name")

def evaluate_policy(model_type):
    """Evaluate the agent for n_eval_episodes."""
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    skip_index=1
    locations=[]
    images=[]
    done=False
    data=defaultdict(lambda :[])
    models={
        "gnm":GNM_Policy,
        "nomad":NoMaD,
        "vint":ViNT_Policy
    }
    default_checkpointts={
        "gnm":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm.pth",
        "nomad":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
        "vint":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth"
    }
    MODEL=models[model_type]
    checkpoint_path=default_checkpointts[model_type]
    # https://github.com/carla-simulator/carla/issues/2832


    if FLAGS.map_dir is None:
        if model_type != "nomad":
            raise ValueError("Only NoMaD can explore")
        agent = MODEL(ckpt_path=checkpoint_path,mode="explore")
    else:
        agent = MODEL(ckpt_path=checkpoint_path,mode="navigate",skip_index=skip_index ,map_dir=FLAGS.map_dir)
    
        with open(f'{FLAGS.map_dir}/aux.pkl', 'rb') as handle:
                dataset=pickle.load(handle)
        topomap_filenames = sorted(glob.glob(f"{os.path.join(FLAGS.map_dir)}/*.jpg"),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        locations=dataset["location"]
        # user_defined_trajectories=[]
        # with open(f'full_traj_aux.pkl', 'wb') as handle:
        #         for _ in range(5):
        #             start,end=random.choices(locations,k=2)
        #             user_defined_trajectories.append([start,end])
        #         pickle.dump(dict(locations=user_defined_trajectories),handle)
        num_nodes = len(topomap_filenames)
        # new_map_image=[]
        new_locations=[]
        path=f"baseline_maps/{model_type}"
        os.makedirs(path,exist_ok=True)
        step=0
        while not done:
             
            for ix in range(0,num_nodes):
                    img_path=topomap_filenames[ix]
                    img=(np.asarray(Image.open(img_path))[...,None]/255).astype(np.float32)
                    # print(img)
                    if ix==0:
                        for _ in range(5):
                            _,_=agent.can_traverse(img)
                    can_traverse,done=agent.can_traverse(img)
                    if not can_traverse:
                        Image.open(img_path).save(f"{path}/{step}.jpg")
                        new_locations.append(locations[ix])
                        step+=1
    

    with open(f"{path}/aux.pkl", "wb") as f:
        pickle.dump(dict(locations=locations,
                         town="Town02"
                         ),f)
    return {}

def main(_):

    # Evaluate
    stats = evaluate_policy(
        FLAGS.model
    )
    
  

if __name__ == "__main__":
    # flags.mark_flag_as_required("checkpoint_path")
    app.run(main)