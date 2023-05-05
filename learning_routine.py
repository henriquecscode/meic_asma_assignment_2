from stable_baselines3 import A2C, PPO
from utils import *
from training_utils import *
# Not working

env = "LunarLander-v2"
suffix = "simple"
TIMESTEPS = 1000
model = PPO("MlpPolicy",env, verbose=1, tensorboard_log=LOGS_DIR)
path = get_model_path(model, env, suffix)
routine = create_training_routine(model, path)
setup_training_interrupt()
routine(10, 10000)

