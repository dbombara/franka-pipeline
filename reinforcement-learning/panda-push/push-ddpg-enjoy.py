import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
import PIL
from stable_baselines3.common.logger import configure

env = make_vec_env('PandaPush-v3',n_envs=1, seed=98)
model = DDPG.load('agent/ddpg_push_agent',env=env)
policy = model.policy
obs = env.reset()

images = []
for i in range(100):
    action, _states = model.predict(obs)
    obs = env.step(action)[0]
    img = env.render()
    img = PIL.Image.fromarray(img)
    images.append(img)
    print("Iteration: ",i)

env.close()

images[0].save('gif/ddpg_push.gif',
              save_all=True, append_images=images[1:],
              optimize=False, duration=40, loop=0)