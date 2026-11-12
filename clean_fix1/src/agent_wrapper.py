# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import copy
import math
from typing import Optional
import numpy as np
import carla
from rlib_integration.helper import get_speed,local2world,ndarray_to_location,carla_location_to_np_array,carla_rotation_to_np_array,draw_waypoints


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(
        self,
        vehicle,
        # args_lateral,
        # args_longitudinal,
        offset=0,
        max_throttle=0.75,
        max_brake=0.3,
        max_steering=0.8,
        dt=0.1
    ):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """
        args_lateral={
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': dt,
        }
        args_longitudinal={
                'K_P': 1.0,
                'K_D': 0,
                'K_I': 1.0,
                'dt': dt,
            }

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering
        self.dt=args_longitudinal['dt']
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(
            self._vehicle, **args_longitudinal
        )
        self._lat_controller = PIDLateralController(
            self._vehicle, offset, **args_lateral
        )

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """
        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return current_steering ,np.clip(acceleration,-self.max_brake,self.max_throt)

    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLateralController"""
        self._lat_controller.change_parameters(**args_lateral)

    def set_offset(self, offset):
        """Changes the offset"""
        self._lat_controller.set_offset(offset)


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print("Current speed = {}".format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        # print("error",error)
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
        )

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def set_offset(self, offset):
        """Changes the offset"""
        self._offset = offset

    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

            :param waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = waypoint.transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(
                x=self._offset * r_vec.x, y=self._offset * r_vec.y
            )
        else:
            w_loc = waypoint.transform.location

        w_vec = np.array([w_loc.x - ego_loc.x, w_loc.y - ego_loc.y, 0.0])

        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            _dot = 1
        else:
            _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        # print("dot",_dot)
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
        )

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt



class SetPointAgent():
    """An agent that predicts setpoints and consumes them with a PID
  controller."""

    def __init__(
            self,
            vehicle,
            *,
            setpoint_index: int = 5,
            replan_every_steps: int = 1,
            fixed_delta_seconds_between_setpoints: Optional[int] = None) -> None:


        # References to the CARLA objects.
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        dt = self._vehicle.get_world().get_settings().fixed_delta_seconds
        # Sets up PID controllers.
        # dt = self._vehicle.get_world().get_settings().fixed_delta_seconds
        # lateral_control_dict = lateral_control_dict.copy()
        # lateral_control_dict.update({"dt": dt})

        # longitudinal_control_dict = longitudinal_control_dict.copy()
        # longitudinal_control_dict.update({"dt": dt})

        self._vehicle_controller = VehiclePIDController(
            vehicle=self._vehicle,dt=dt
        )

        # Sets agent's hyperparameters.
        self._setpoint_index = setpoint_index
        self._replan_every_steps = replan_every_steps
        self._fixed_delta_seconds_between_setpoints = fixed_delta_seconds_between_setpoints or self._vehicle_controller.dt
        assert self._fixed_delta_seconds_between_setpoints==dt
        # Inits agent's buffer of setpoints.
        self._setpoints_buffer = None
        self._steps_counter = 0


    def run_step(self,waypoints):

        # Current measurements used for local2world2local transformations.
        # waypoints=np.array([*waypoints.tolist(),0.0])
        waypoints=np.insert(waypoints,2,0,axis=-1)
        transform=self._vehicle.get_transform()
        current_location = carla_location_to_np_array(transform.location)
        current_rotation = carla_rotation_to_np_array(transform.rotation)

        # if self._setpoints_buffer is None or self._steps_counter % self._replan_every_steps == 0:
            # Get agent predictions.
            # Transform plan to world coordinates
        predicted_plan_world = local2world(
            current_location=current_location,
            current_rotation=current_rotation,
            local_locations=waypoints,
        )

        # Refreshes buffer.
        self._setpoints_buffer = predicted_plan_world
        # breakpoint()
        # print(self._setpoints_buffer)
        # else:
        #     # Pops first setpoint from the buffer.
        #     self._setpoints_buffer = self._setpoints_buffer

        # Registers setpoints for rendering.
        # self._environment.unwrapped.simulator.sensor_suite.get(
        #     "predictions").predictions = world2local(
        #     current_location=current_location,
        #     current_rotation=current_rotation,
        #     world_locations=self._setpoints_buffer,
        # )
        # breakpoint()
        # Increments counter.
        self._steps_counter += 1
        # Calculates target speed by averaging speed in the `setpoint_index` window.
        target_speed = np.linalg.norm(
            np.diff(self._setpoints_buffer, axis=0),
            axis=1,
        ).mean() / (self._fixed_delta_seconds_between_setpoints)

        # Converts plan to PID controller setpoint.
        # breakpoint()
        setpoint = self._map.get_waypoint(ndarray_to_location(self._setpoints_buffer[0]),project_to_road=False)
        # draw_waypoints( self._world,[setpoint])
        # print("waypoint",np.arctan2(waypoints[1],waypoints[0]))
        # Avoids getting stuck when spawned.
        # if self._steps_counter <= 100:
        #     target_speed = 20.0 / 3.6
        if setpoint is None:
            return 0.0 ,0.0
        # Run PID step.
        control = self._vehicle_controller.run_step(
            target_speed=target_speed*3.6,  # PID controller expects speed in km/h!
            waypoint=setpoint,
        )
        return control