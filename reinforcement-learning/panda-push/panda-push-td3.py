import gymnasium as gym
import panda_gym


from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
import PIL
from stable_baselines3.common.logger import configure

env = make_vec_env('PandaPush-v3',n_envs=1, seed=98)

tmp_path = "/tmp/sb3_log/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model = TD3('MultiInputPolicy',
            env,
            batch_size=2048,
            buffer_size=1000000,
            gamma=0.95,
            learning_rate=0.001,
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            #replay_buffer_class='HerReplayBuffer',
            #replay_buffer_kwargs= dict(online_sampling=True, goal_selection_strategy='future',n_sampled_goal=4),
            tau=0.05,
            verbose=1)

model.set_logger(new_logger)

model.learn(total_timesteps=1000000.0)

model.save('agent/td3_stablebaselines_agent')
policy = model.policy
policy.save("policy/td3_policy_pandapush")
model = TD3.load('agent/td3_stablebaselines_agent')
obs = env.reset()

images = []
for i in range(50):
    action, _states = model.predict(obs)
    obs = env.step(action)[0]
    img = env.render()
    img = PIL.Image.fromarray(img)
    images.append(img)
    print("Iteration: ",i)

env.close()

images[0].save('gif/td3_stablebaselines_example.gif',
              save_all=True, append_images=images[1:],
              optimize=False, duration=40, loop=0)