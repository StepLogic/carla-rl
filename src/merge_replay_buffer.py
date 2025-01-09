import glob
import pickle
from train_online_pixels import CarlaGoalEnv,config,FrameStack,TimeLimit,RecordEpisodeStatistics,ReplayBuffer

for path in glob.glob("datasets/*.pkl"):
    env = CarlaGoalEnv(config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        int(1e5)
    )

    with open(path, 'rb') as handle:
            dataset=pickle.load(handle)
    for entry in dataset:
          replay_buffer.insert(entry)
          