from stable_baselines3 import A2C
from utils import *
# Not working

env = "LunarLander-v2"
suffix = "simple"
TIMESTEPS = 1000
model = A2C("MlpPolicy",env, verbose=1, tensorboard_log=LOGS_DIR)
path = get_model_path(model, env, suffix)
model.learn(TIMESTEPS, tb_log_name=path)

save_model_to(model, path)
