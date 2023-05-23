from stable_baselines3 import A2C, PPO
from utils import *
from training_utils import *
from modified_env_utils import get_env_name_from_params
import sys
import math
# Not working

EPISODES = 15
ITERATIONS = 10000

filepath = ""
iterations = 0
if len(sys.argv) > 1:
    family_name = sys.argv[1]
    model, env, family_name, iterations = get_model_env(family_name)
    path = family_name
else:
    params = {}
    env = get_wrapped_lunar_environment(**params)
    env_name = get_env_name_from_params(params)
    suffix = "simple"
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOGS_DIR)
    path = get_model_path(model, env_name, suffix)
    iterations = 0


episodes = None
episodeIterations = None
totalIterations = None
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        # Args are of the type "--key=value"
        key, value = arg.split("=")
        if key == "--episodes":
            episodes = int(value)
        elif key == "--iterations":
            episodeIterations = int(value)
        elif key == "--totalIterations":
            totalIterations = int(value) - iterations

if totalIterations is not None:
    if totalIterations <= 0: 
        print("Model already fully trained.")
        exit(0)
    if episodeIterations is None:
        episodeIterations = ITERATIONS
    episodes = math.ceil(totalIterations // episodeIterations)
else:
    if episodes is None:
        episodes = EPISODES
    if episodeIterations is None:
        episodeIterations = ITERATIONS

routine = create_training_routine(model, path, iterations)
print(f"Retraining model from {path} starting on iteration {iterations}\nWill train for {episodes} episodes of {episodeIterations} iterations each")
routine(episodes, episodeIterations)
