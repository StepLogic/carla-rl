#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import collections.abc
import math
import os
import shutil

import carla
import cv2
import numpy as np

from tensorboard import program


def post_process_image(image, normalized=True, grayscale=True,crop=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: image
    """
    # crop the sky
    if crop:
        image = image[100:, :, :]
    image = cv2.resize(image, (84, 84))

    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.uint8)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def find_latest_checkpoint(directory):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """
    start = directory
    max_checkpoint_int = -1
    checkpoint_path = ""

    # 1st layer: Check for the different run folders
    for f in os.listdir(start):
        if os.path.isdir(start + "/" + f):
            temp = start + "/" + f

            # 2nd layer: Check all the checkpoint folders
            for c in os.listdir(temp):
                if "checkpoint_" in c:

                    # 3rd layer: Get the most recent checkpoint
                    checkpoint_int = int(''.join([n for n in c
                                                  if n.isdigit()]))
                    if checkpoint_int > max_checkpoint_int:
                        max_checkpoint_int = checkpoint_int
                        checkpoint_path = temp + "/" + c + "/" + c.replace(
                            "_", "-")

    if not checkpoint_path:
        raise FileNotFoundError(
            "Could not find any checkpoint, make sure that you have selected the correct folder path"
        )

    return checkpoint_path


def get_checkpoint(name, directory, restore=False, overwrite=False):
    training_directory = os.path.join(directory, name)

    if overwrite and restore:
        raise RuntimeError(
            "Both 'overwrite' and 'restore' cannot be True at the same time")

    if overwrite:
        if os.path.isdir(training_directory):
            shutil.rmtree(training_directory)
            print("Removing all contents inside '" + training_directory + "'")
        return None


    if restore:
        return find_latest_checkpoint(training_directory)

    if os.path.isdir(training_directory) and len(os.listdir(training_directory)) != 0:
        raise RuntimeError(
            "The directory where you are trying to train (" +
            training_directory + ") is not empty. "
            "To start a new training instance, make sure this folder is either empty, non-existing "
            "or use the '--overwrite' argument to remove all the contents inside"
        )

    return None


def launch_tensorboard(logdir, host="localhost", port="6006"):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--host", host, "--port", port])
    url = tb.launch()
def carla_location_to_np_array(location):
    return np.array([location.x, location.y, location.z])

import collections
from dataclasses import dataclass
import heapq
from scipy.optimize import fsolve, root
from scipy.interpolate import splprep, splev
import numpy as np
from scipy.optimize import fsolve,root     
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve,root

def make_curve(points):
    t = np.linspace(0, 1, len(points))
    tckp, u = splprep(points.T, u=t,k=3 )
    return lambda t:np.array(splev(t,tckp)),tckp

def derivative_curve(tckp):
    return lambda t:np.array(splev(t,tckp,der=1))

def get_curve(points):
        indx = np.argsort(points[:,0])
        # print(points,points.shape)
        points = points[indx]
        f_t,tckp=make_curve(points)
        f_prime_t=derivative_curve(tckp)
        t_samples = np.linspace(0, 1, 1000)
        curve_points = np.array([f_t(t) for t in t_samples])
        def curve(x):
            func=lambda v:np.dot((x-f_t(v[-1])).T,f_prime_t(v[-1]))
            t_initial=0.5
            # result=root(func,[t_initial],method="anderson")
            # if result.success:
            #     t=result.x.squeeze()
            #     return f_t(t),f_prime_t(t)
            distances = np.linalg.norm(curve_points - x, axis=1)
            closest_idx = np.argmin(distances)
            t = t_samples[closest_idx]
            return f_t(t),f_prime_t(t)
        return curve
def draw_waypoints(world, waypoints, z=0.5, lifetime=-1.0, color=(255,0,0)):
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        # Increase the length of the arrow for better visibility
        length = 1.0  # Adjust this value to change arrow length
        end = begin + carla.Location(
            x=length * math.cos(angle), 
            y=length * math.sin(angle)
        )
        
        # Add forward vector normalization
        forward_vector = end - begin
        forward_vector = forward_vector.make_unit_vector() * length
        end = begin + forward_vector
        
        # Ensure arrow is visible above ground
        begin.z += 0.5  # Adjust height offset as needed
        end.z = begin.z  # Keep arrow parallel to ground
        
        # Draw the arrow with debug helper
        world.debug.draw_arrow(
            begin=begin,
            end=end,
            thickness=0.1,  # Add thickness parameter
            arrow_size=0.3,
            color=carla.Color(*color),
            life_time=lifetime
        )