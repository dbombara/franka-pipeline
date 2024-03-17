import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
import PIL
from stable_baselines3.common.logger import configure

env = make_vec_env('PandaPush-v3',n_envs=1, seed=98)

tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = DDPG('MultiInputPolicy',
            env,
            verbose=0,
            replay_buffer_class=HerReplayBuffer)

model.set_logger(new_logger)

model.learn(total_timesteps=1e5)

model.save('rl_models/ddpg_stablebaselines_agent')
policy = model.policy
policy.save("rl_policies/ddpg_policy_pandapush")