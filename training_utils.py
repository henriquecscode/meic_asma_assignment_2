
from utils import *
from stable_baselines3.common.base_class import BaseAlgorithm
import signal

EPISODES = 10
TIMESTEPS = 1000

stop_training = False

def handler(signum, frame):
    global stop_training
    stop_training = True
    print("Stopping training due to interrupt")

def setup_training_interrupt():
    signal.signal(signal.SIGINT, handler)

def create_training_routine(model: BaseAlgorithm, path=""):
    if path == "":
        path = get_model_path(model, "", "")

    def learning_routine(episodes = EPISODES, timesteps = TIMESTEPS):
        # Use a negative number to run forever
        episode_count = 1
        while(episodes != 0 and not stop_training):
            model.learn(timesteps, reset_num_timesteps=False, tb_log_name=path)
            episode_path = f"{path}/{timesteps*episode_count}"
            save_model_to(model, episode_path)
            episodes -= 1
            episode_count += 1

        print("Stopping training routine")


    return learning_routine