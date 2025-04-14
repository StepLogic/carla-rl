#!/usr/bin/env python

from __future__ import print_function
import gymnasium as gym
from rlib_integration.carla_core import CarlaCore
from rlib_integration.helper import carla_location_to_np_array,carla_rotation_to_np_array
from configs.baseline_env_config import config ,JAXMappingExperiments
import carla
class CarlaEvalEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """
    def __init__(self,use_rgb=False,image_size=64,start_server=True,town="Town01",max_dist=100):
        """Initializes the environment"""
        self.config = config["env_config"]
        self.experiment = JAXMappingExperiments(self.config["experiment"],is_rgb=use_rgb,image_size=image_size)
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        sim_conf=self.config['carla']
        sim_conf.update({
            "max_dist":max_dist,
            "start_server":start_server,
            "town":town
        })
        self.max_dist=max_dist
        # print(sim_conf,self.config["experiment"])
        self.core = CarlaCore(sim_conf)
        self.core.setup_experiment(self.experiment.config)
        self.last_position=None
        # self.reset()
    def set_start_transform(self,start):
        # print(start)
        self.experiment.origin=self.core.map.get_waypoint(start)

    def set_destination_transform(self,end):
        self.experiment.destination=self.core.map.get_waypoint(end)
    def set_goal(self,goal_image,heading,location=None):
        self.experiment.set_goal(goal_image,heading,location=location)

    def is_agent_at_junction(self):
        wp =self.core.map.get_waypoint(self.hero.get_transform().location,project_to_road=True)
        return wp.is_junction,carla_location_to_np_array(self.hero.get_transform().get_right_vector()),carla_location_to_np_array(self.hero.get_transform().location)  
            
    def reset(self,*arg,**kwargs):
        # Reset sensors hero and experiment
        self.experiment.reset(self)
        self.experiment.config["hero"]["is_goal_env"]=True
        if self.experiment.origin and self.experiment.destination:
            self.experiment.config["hero"]["origin"]=self.experiment.origin.transform or self.last_position 
            self.experiment.config["hero"]["destination"]=self.experiment.destination.transform or self.last_position 
            self.hero = self.core.reset_hero_for_experiments(self.experiment.config["hero"])
        else:
            if not  self.experiment.trajectories is None:
                self.experiment.config["hero"]["trajectory"]=self.experiment.trajectories
            if hasattr(self.experiment,"curriculum_step"):
                self.experiment.config["hero"]["curriculum_step"]=self.experiment.curriculum_step
            self.hero = self.core.reset_hero(self.experiment.config["hero"],max_dist=self.max_dist)
        if hasattr(self.experiment,"curriculum_step"):
            self.experiment.config["hero"]["curriculum_step"]=self.experiment.curriculum_step
  
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
        self.last_position=self.core.map.get_waypoint(self.core.hero.get_transform().location).transform
        return observation, reward, done,False,self.experiment.info