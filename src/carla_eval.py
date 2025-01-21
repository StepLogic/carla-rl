#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function
import gymnasium as gym
from rlib_integration.carla_core import CarlaCore
from src.jax_mapping_experiment import JAXMappingExperiments
import carla
class CarlaEvalEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """
    def __init__(self, config,use_rgb=False,image_size=32):
        """Initializes the environment"""
        self.config = config
        self.experiment = JAXMappingExperiments(self.config["experiment"],is_rgb=use_rgb,image_size=image_size)
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        self.core = CarlaCore(self.config['carla'],map_env=True)
        self.core.setup_experiment(self.experiment.config)
        self.reset()
    def set_start_transform(self,start):
        self.experiment.origin=self.core.map.get_waypoint(start)
    def set_goal(self,goal_image,heading):
        # breakpoint()
        self.experiment.set_goal(goal_image,heading)

    def is_agent_at_junction(self):
        wp =self.core.map.get_waypoint(self.hero.get_transform().location,project_to_road=True)
        return wp.is_junction  
            
    def reset(self,*arg,**kwargs):
        # Reset sensors hero and experiment
        self.experiment.reset(self)
        # breakpoint()
        self.experiment.config["hero"]["is_goal_env"]=True
        self.experiment.config["hero"]["origin"]=self.experiment.origin
        if hasattr(self.experiment,"curriculum_step"):
            self.experiment.config["hero"]["curriculum_step"]=self.experiment.curriculum_step
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        
        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data, self.core)
        return observation ,self.experiment.info

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)
        observation, info = self.experiment.get_observation(sensor_data, self.core)
        done = self.experiment.get_done_status(sensor_data, self.core)
        reward = self.experiment.compute_reward(sensor_data, self.core)
        return observation, reward, done,False,self.experiment.info