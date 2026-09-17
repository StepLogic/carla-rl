#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import cv2 
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import pickle
import math

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# Functional stuff

image_count=0
save_dir="/home/robotlab/scratch/carla-rl/topomap"
locations=[]
os.makedirs(save_dir, exist_ok=True)
def draw_image(surface, image,vehicle=None ,blend=False):
    global image_count
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
 
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if not vehicle is None:
        cv2.imwrite(f"{save_dir}/{image_count}.jpg",cv2.cvtColor(array,cv2.COLOR_BGR2RGB))    
        image_count+=1
        loc=vehicle.get_transform().location
        locations.append([loc.x,loc.y,loc.z])
        with open(f"{save_dir}/traj.pkl", "wb") as pipe:
            pickle.dump({"locations": locations}, pipe)
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

# def filter_roads(wp):
    

def main():
    actor_list = []
    pygame.init()
    
    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world("Town02")
    
    try:
        m = world.get_map()
        waypoints = m.generate_waypoints(2)
        start_pose = waypoints[0].transform
        start_pose.location.z += 5
        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(
            blueprint_library.filter('vehicle.*')[0],
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.0, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            waypoint_index = 0
            for waypoint in waypoints:
                if should_quit():
                    return
                clock.tick()
                
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                
                # Ensure proper forward vector alignment
                transform = waypoint.transform
                
                # Get the forward vector from the waypoint
                forward_vector = transform.get_forward_vector()
                
                # Calculate yaw angle from forward vector
                import math
                yaw = math.atan2(forward_vector.y, forward_vector.x) * 180.0 / math.pi
                
                # Create new transform with proper rotation
                new_transform = carla.Transform(
                    transform.location,
                    carla.Rotation(pitch=0, yaw=yaw, roll=0)
                )
                
                # Alternative method: Use the waypoint's rotation directly (recommended)
                # The waypoint already contains the correct forward-facing rotation
                vehicle.set_transform(waypoint.transform)
                
                # If you need to get the next waypoint in sequence:
                if waypoint_index < len(waypoints) - 1:
                    next_waypoint = waypoints[waypoint_index + 1]
                    # Calculate direction to next waypoint
                    current_loc = waypoint.transform.location
                    next_loc = next_waypoint.transform.location
                    
                    # Calculate forward direction
                    direction = carla.Vector3D(
                        next_loc.x - current_loc.x,
                        next_loc.y - current_loc.y,
                        0  # Keep Z as 0 for ground movement
                    )
                    
                    # Calculate yaw from direction vector
                    if direction.x != 0 or direction.y != 0:
                        yaw = math.atan2(direction.y, direction.x) * 180.0 / math.pi
                        oriented_transform = carla.Transform(
                            waypoint.transform.location,
                            carla.Rotation(pitch=0, yaw=yaw, roll=0)
                        )
                        vehicle.set_transform(oriented_transform)
                
                waypoint_index += 1
                
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                # Draw the display.
                draw_image(display, image_rgb, vehicle)
                draw_image(display, image_semseg, None, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
    
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')