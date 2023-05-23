from utils import save_model_to, get_model_path
from stable_baselines3.common.base_class import BaseAlgorithm
import signal
import datetime

EPISODES = 10
TIMESTEPS = 1000

stop_training = False

def handler(signum, frame):
    global stop_training
    stop_training = True
    print("Stopping training due to interrupt")

def setup_training_interrupt():
    signal.signal(signal.SIGINT, handler)


def  create_training_routine(model: BaseAlgorithm, path="", start=0):
    setup_training_interrupt()
    if path == "":
        path = get_model_path(model, "", "")

    def learning_routine(episodes=EPISODES, timesteps=TIMESTEPS):
        # Use a negative number to run forever
        episode_count = 1
        total_episodes = episodes
        while (episodes != 0 and not stop_training):
            episode_start = datetime.datetime.now()
            model.learn(timesteps, reset_num_timesteps=False, tb_log_name=path)
            episode_end = datetime.datetime.now()
            duration = episode_end - episode_start
            duration = duration - datetime.timedelta(microseconds=duration.microseconds)
            episode_path = f"{path}/{start + timesteps*episode_count}"
            save_model_to(model, episode_path)
            print(f"Finished {episode_count}/{total_episodes} episodes for {model.__class__.__name__} in {duration} in {path}")
            episodes -= 1
            episode_count += 1

        print("Finished training routine")

    return learning_routine

