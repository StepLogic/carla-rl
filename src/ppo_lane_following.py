import glob
import os
import time
from collections import defaultdict, deque
from jaxrl2.utils.misc import Logger
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import ml_collections
import numpy as np
import tqdm
from src.configs.train_env_config import config as carla_config
from rlib_integration.carla_goal_env import CarlaGoalEnv
from flax.training import checkpoints
import flax
flax.config.update('flax_use_orbax_checkpointing', True)


def save_checkpoint(agent, path, step):
    os.makedirs(path, exist_ok=True)
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'rng': agent._rng,
    }
    checkpoints.save_checkpoint(
        ckpt_dir=os.path.abspath(path),
        target=state_dict,
        step=step,
        overwrite=True,
        keep=3  # Keep last 3 checkpoints
    )
# Main function
def main():
    # Training parameters
    MAX_STEPS = int(5e6)
    EVAL_INTERVAL = int(5e4)
    EVAL_EPISODES = 5
    BATCH_SIZE = 64
    SEED = 42
    LOCAL_STEPS=2048

    # Create environment
    env = CarlaGoalEnv(carla_config["env_config"])
    env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    env = TimeLimit(env,max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)


    env.reset(seed=SEED)
    np.random.seed(SEED)

    # Initialize logger
    logger = Logger(log_dir="./logs",prefix="PPO")

    # Initialize checkpoint-directory 
    policy_folder = os.path.join("checkpoints", f"model-ppo-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)


    # Initialize PPO agent
    from jaxrl2.agents import PPOLearner
    config = ml_collections.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "d4pg"
    config.discount = 0.98
    config.critic_reduction = "mean"
    config.clip_ratio = 0.2  # Add clip ratio
    config.gae_lambda = 0.95  # Add GAE lambda
    config = config.to_dict()

    agent = PPOLearner(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        **config
    )

    # Initialize replay buffer
    from jaxrl2.data import RolloutBuffer
    replay_buffer = RolloutBuffer(
        env.observation_space,
        env.action_space,
        capacity=LOCAL_STEPS  # Store 100 batches worth of data
    )
    # replay_buffer.seed(SEED)

    success_history = deque(maxlen=100)  # Track last 100 episodes
    eval_success_history = deque(maxlen=100)

    distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    eval_distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    # Training loop
    observation, info = env.reset()
    episode_return = 0
    episode_length = 0
    training_start_time = time.time()
    # n_steps=int(ROLLOUT_CAPACITY/2)  #rollout steps
    n_updates=0
    p_bar = tqdm.tqdm(range(MAX_STEPS))
    p_bar.update(5)
    p_bar.refresh()
    for step in range(1, MAX_STEPS + 1,LOCAL_STEPS):
        n_step=0
        while n_step < LOCAL_STEPS:
            action, logp, value = agent.sample_actions(observation)
            # print(value)
            # action = np.clip(action, env.action_space.low, env.action_space.high)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            timeout = "TimeLimit.truncated" in info
            mask = 1.0 if not done or timeout else 0.0
            # Store transition in replay buffer
            # print(reward)
            if done:
                if timeout:
                    _,_,last_value = agent.sample_actions(next_observation)
                    reward = reward + config.get("discount",0.99) * last_value
                # else:
            #     #     last_value = 0.0  # Terminal state has value of 0
            # # print(n_step)
            # print("after",reward)
            replay_buffer.insert(
                dict(
                    observations=observation,
                    logps=logp,
                    values=value,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                )
            )
            observation = next_observation
            episode_return += reward
            episode_length += 1
            episode_info = {
                    "return": episode_return,
                    "length": episode_length,
                }
            if done:
                if "is_success" in info:
                    success = float(info["is_success"])
                    success_history.append(success)
                    episode_info["is_success"] = success
                    episode_info["success_rate"] = np.mean(success_history)
                if "distance_completed" in info:
                    distance_completed = float(info["distance_completed"])
                    distance_to_goal_history.append(distance_completed)
                    episode_info["distance_completed"] = distance_completed
                    episode_info["distance_completed"] = np.mean(distance_to_goal_history)
                episode_info = {
                    "return": episode_return,
                    "length": episode_length,
                    "mean_reward":info.get("mean_reward",None),
                    "max_reward":info.get("max_reward",None),
                    "min_reward":info.get("min_reward",None)
                }

                logger.log_episode(episode_info, step)
                observation, info = env.reset()
                episode_return = 0
                episode_length = 0
            n_step+=1
            p_bar.n = n_step+step
            p_bar.refresh()


        _,_,last_value = agent.sample_actions(next_observation)
        replay_buffer.compute_advantage(last_value=last_value,done=done,gae_lambda=0.95,discount=config.get("discount",0.99))
        
        # print(replay_buffer.can_sample(),len(replay_buffer),replay_buffer._path_start_idx)
        if len(replay_buffer) >= BATCH_SIZE:
            total_metrics = defaultdict(list)
            num_updates = 0
            for _ in range(10):  # 10 epochs
                replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": BATCH_SIZE})
                for batch in replay_buffer_iterator:
                    update_info = agent.update(batch, utd_ratio=1)
                    for key, value in update_info.items():
                        total_metrics[key].append(float(value))
                    num_updates += 1
            n_updates += 10
            # Calculate averages for each metric
            average_metrics = {
                key: np.mean(value)  
                for key, value in total_metrics.items()
            }
            average_metrics["explained_variance"]=replay_buffer.explained_variance
            average_metrics["total_updates"]=n_updates
            average_metrics["total_timesteps"]=step
            # print(average_metrics,total_metrics)
            logger.log_training(average_metrics, step)
            logger.print_status(step, MAX_STEPS)
            



            # Periodic evaluation
        if step % EVAL_INTERVAL == 0:
                eval_returns = []
                eval_lengths = []
                eval_successes = []
                eval_dists = []
                eval_mean_reward = []
                eval_min_reward = []
                eval_max_reward = []
                for _ in range(EVAL_EPISODES):
                    eval_obs, _ = env.reset()
                    eval_done = False
                    eval_return = 0
                    eval_length = 0

                    while not eval_done:
                        eval_action = agent.eval_actions(eval_obs)
                        eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = env.step(eval_action)
                        eval_done = eval_terminated or eval_truncated
                        eval_return += eval_reward
                        eval_length += 1

                    eval_returns.append(eval_return)
                    eval_lengths.append(eval_length)
                    if "is_success" in eval_info:
                        eval_successes.append(float(eval_info["is_success"]))
                    if "distance_completed" in eval_info:
                        eval_dists.append(float(eval_info["distance_completed"]))


                    eval_mean_reward.append(info.get("mean_reward",0))
                    eval_max_reward.append(info.get("max_reward",0))
                    eval_min_reward.append(info.get("min_reward",0))
                

                eval_info = {
                    "eval_return_mean": np.mean(eval_returns),
                    "eval_return_std": np.std(eval_returns),
                    "eval_length_mean": np.mean(eval_lengths),
                    "eval_length_std": np.std(eval_lengths),
                    "eval_mean_reward_per_step":np.mean(eval_mean_reward),
                    "eval_max_reward_per_step":np.mean(eval_max_reward),
                    "eval_min_reward_per_step":np.mean(eval_min_reward)

                }
                save_checkpoint(agent,policy_folder,i)
                logger.log_eval(eval_info, step)
                logger.print_status(step, MAX_STEPS)

    # Print final training statistics
    save_checkpoint(agent,f"checkpoints/final_ppo",1)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")
    # Close environments
    env.close()

if __name__ == "__main__":
    main()