import os
from typing import Callable

import gymnasium
import gc
import cv2
from moviepy.editor import *


class VideoRecorder(gymnasium.Wrapper):

    def __init__(
        self,
        venv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 1000,
        name_prefix: str = "rl-video",
    ):
        gymnasium.Wrapper.__init__(self, venv)

        self.env = venv
        # self.env.render_mode="rgb_array"
        self.record_video_trigger = record_video_trigger
        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)
        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length
        self.recording = False
        self.recorded_frames = 0
        self.frames = {}
        self.length = video_length

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        if len(self.frames.keys())>100:
            self.store_videos()
        # self.frames = {}
        return obs

    def step(self, action):
        # breakpoint()
        if self.record_video_trigger(self.step_id):
            # print("he",self.step_id)
            if self.step_id > 0 and self.step_id % self.length == 0:
                self.store_videos()
                self.frames = {}
            self.frames[self.step_id % self.length] = self.env.render()
        self.step_id += 1
        return self.env.step(action)

    def store_videos(self) -> None:

        video_name = f"{self.name_prefix}"
        # video_name = f"{self.name_prefix}"
        base_path = os.path.join(self.video_folder, video_name)

        # size = (320,200)

        # out = cv2.VideoWriter(f'{base_path}.mp4',cv2.VideoWriter_fourcc(*'DIVX'),15, size)

        # for i in range(len(list(self.frames.values()))):
        #     rgb_img = cv2.cvtColor(list(self.frames.values())[i], cv2.COLOR_RGB2BGR)
        #     out.write(rgb_img)
        # out.release()
        clip = ImageSequenceClip(list(self.frames.values()), fps=10)
        # print("len",list(self.frames.values()))
        clip.write_videofile(f"{base_path}.mp4", verbose=False, logger=None)
        clip.close()
        del clip
        gc.collect()
