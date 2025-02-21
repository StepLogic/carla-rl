import rospy
from src.leo.leo_env import LeoEnv
import numpy as np
import cv2
rospy.init_node("PD_CONTROLLER", anonymous=False)
env=LeoEnv()
observation,info=env.reset()
res=cv2.imwrite("test.jpg",observation["pixels"])
# print(observation["vector"],res)
while not rospy.is_shutdown():
    next_observation, reward, done, truncated, info=env.step(env.action_space.sample())
    # cv2.imwrite("test.jpg",next_observation["pixels"])

    # print(next_observation["vector"])