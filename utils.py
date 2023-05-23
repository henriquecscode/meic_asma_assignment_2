import datetime
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from modified_env import get_wrapped_lunar_environment, get_environment_action_space, get_environment_observation_space
from modified_env_utils import get_param_from_env_name
import os
MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'
MODELS = [A2C, DDPG, PPO, SAC, TD3]

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


def get_date():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_model_path(model: BaseAlgorithm, env_name: str, suffix: str = ""):
    return f"{get_date()}_{env_name}_{get_model_type_name(model)}{ '_'+suffix if suffix != '' else ''}"


def get_model_type_name(model: BaseAlgorithm):
    return model.__class__.__name__


def save_model(model: BaseAlgorithm, env_name, suffix=""):
    filepath = get_model_path(model, env_name, suffix)
    model.save(filepath)


def save_model_to(model: BaseAlgorithm, file):
    filepath = f"{MODELS_DIR}{file}"
    model.save(filepath)


def load_model(filename, env=None):
    filepath = f"{MODELS_DIR}{filename}"
    model_type = get_model_type(filepath)
    for model in MODELS:
        if model.__name__ == model_type:
            cls: BaseAlgorithm = model
            break
    try:
        return cls.load(filepath, env)
    except KeyError as e:
            action_space = get_environment_action_space(env)
            observation_space = get_environment_observation_space(env)
            custom_objects = {
                "action_space": action_space,
                "observation_space": observation_space
            }
            return cls.load(filepath, env, custom_objects=custom_objects)
        


def get_model_type(filepath):
    filepath_parts = filepath.split("/")
    model_path = filepath_parts[-2]
    model_path_parts = model_path.split("_")
    model_type = model_path_parts[-2]
    return model_type


def get_env(filename):
    filename_parts = filename.split("_")
    env_name = filename_parts[1]
    return env_name

def use_model(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()


def get_models():
    return MODELS

def get_iterations_from_model_name(filepath):
    filepath_parts = filepath.split(".")
    iterations = filepath_parts[0]
    iterations = int(iterations)
    return iterations

def get_data_from_path(filepath):
    filepath_parts = filepath.split("/")
    family_name = filepath_parts[0]
    if len(filepath_parts) > 1:
        # Has iterations
        iterations_parts = filepath_parts[-1].split(".")
        iterations = int(iterations_parts[0])
    else:
        # Get maximum iterations from folder
        # Open directory
        # Get all files
        models_available = os.listdir(f"{MODELS_DIR}{family_name}")
        models_available_iterations = [get_iterations_from_model_name(x)  for x in models_available]
        models_available_iterations.sort()
        iterations = models_available_iterations[-1]
    filename = f"{family_name}/{iterations}"
    return filename, family_name, iterations

def get_model_env(filepath) :
    filename, family_name, iterations = get_data_from_path(filepath)
    env_name = get_env(family_name)
    env_params = get_param_from_env_name(env_name)
    env = get_wrapped_lunar_environment(**env_params)
    model = load_model(filename, env)
    return model, env, family_name, iterations
