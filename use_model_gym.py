
import gym
from utils import *
import sys

DEFAULT_MODEL = "2023_05_05_08_22_47_LunarLander-v2_PPO_simple/100000.zip"
if __name__ == '__main__':
    if len(sys.argv) == 1:
        filename = DEFAULT_MODEL
    else:
        filename = sys.argv[1]

    env_name = get_env(filename)

    env = gym.make(env_name)
    model = load_model(filename)

    obs = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()
