#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""DQN Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from __future__ import print_function

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import argparse
import os
import yaml

import ray
from ray import tune
import os
from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers

from vision_rl.rllib_integration.helper import get_checkpoint, launch_tensorboard

from vision_rl.sac_example.sac_experiment import SACExperiment
from vision_rl.sac_example.sac_callbacks import SACCallbacks
from vision_rl.sac_example.sac_trainer import CustomSACTrainer
# from dqn_example.dqn_experiment import DQNExperiment
# from dqn_example.dqn_callbacks import DQNCallbacks
# from dqn_example.dqn_trainer import CustomDQNTrainer

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperiment
os.environ["CARLA_ROOT"]='/home/robotlab/Apps/CARLA_0.9.15'

def run(args):
    try:
        ray.init(
            address="auto" if args.auto else None,
            ignore_reinit_error=True,
            local_mode=False,
            include_dashboard=True
        )

        tune.run(
            CustomSACTrainer,
            name=args.name,
            local_dir=args.directory,
            stop={
                "perf/ram_util_percent": 85.0,
                "timesteps_total": args.stop_timesteps
            },
            checkpoint_freq=1,
            checkpoint_at_end=True,
            restore=get_checkpoint(args.name, args.directory,args.restore, args.overwrite),
            config=args.config,
            max_failures=-1,
            keep_checkpoints_num=5,
            reuse_actors=False,
            fail_fast=False,
            # max_retries=3,
            # retry_on_error=True,
            sync_config=tune.SyncConfig(
                sync_on_checkpoint=True,
                sync_period=300  # sync every 5 minutes
            ),
            resume="AUTO",  # Automatically detect if experiment exists and resume
            raise_on_failed_trial=False
        )
    finally:
        kill_all_servers()
        ray.shutdown()


def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["callbacks"] = SACCallbacks

    return config


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("-d", "--directory",
                           metavar='D',
                           default=os.path.expanduser("~") + "/ray_results/carla_rllib",
                           help="Specified directory to save results (default: ~/ray_results/carla_rllib")
    argparser.add_argument("-n", "--name",
                           metavar="N",
                           default="dqn_example",
                           help="Name of the experiment (default: dqn_example)")
    argparser.add_argument("--restore",
                           action="store_true",
                           default=False,
                           help="Flag to restore from the specified directory")
    argparser.add_argument("--overwrite",
                           action="store_true",
                           default=True,
                           help="Flag to overwrite a specific directory (warning: all content of the folder will be lost.)")
    argparser.add_argument("--tboff",
                           action="store_true",
                           default=False,
                           help="Flag to deactivate Tensorboard")
    argparser.add_argument("--auto",
                           action="store_true",
                           default=False,
                           help="Flag to use auto address")
    argparser.add_argument("--stop-timesteps",
                           default=100000,
                           type=int,
                           help="Number of timesteps to train for")


    args = argparser.parse_args()
    args.config = parse_config(args)

    if not args.tboff:
        launch_tensorboard(logdir=os.path.join(args.directory, args.name),
                           host="0.0.0.0" if args.auto else "localhost")

    run(args)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
