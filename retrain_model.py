from stable_baselines3 import A2C, PPO
import gym
from utils import *
from training_utils import *
import sys
import os
# Not working

filepath = ""
iterations = 0
if len(sys.argv) > 1:
    family_name = sys.argv[1]
    if len(sys.argv) > 2:
        iterations = sys.argv[2]
        filepath = f"{family_name}/{iterations}"
    else:
        models_available = os.listdir(f"{MODELS_DIR}/{family_name}")
        # Get maximum iterations from folder
        # Open directory
        # Get all files
        # Get all files with .zip
        models_available.sort()
        filepath = models_available[-1]
        iterations = get_iterations(filepath)
        filepath = f"{family_name}/{iterations}"


    env_name = get_env(filepath)
    env = gym.make(env_name)
    model = load_model(filepath, env)
    path = family_name
else:
    env = "LunarLander-v2"
    suffix = "simple"
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR)
    path = get_model_path(model, env, suffix)

routine = create_training_routine(model, path, iterations)
setup_training_interrupt()
routine(10, 10000)
