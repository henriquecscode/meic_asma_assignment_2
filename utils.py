import datetime
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from modified_env import get_wrapped_lunar_environment
import os
MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'

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
    if model_type == "A2C":
        cls = A2C
    elif model_type == "DDPG":
        cls = DDPG
    elif model_type == "PPO":
        cls = PPO
    elif model_type == "SAC":
        cls = SAC
    elif model_type == "TD3":
        cls = TD3

    return cls.load(filepath, env)


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


def get_float_to_str(x: float):
    return str(x).replace(".", "-")


def get_str_to_float(x: str):
    return float(x.replace("-", "."))


def get_env_name_from_params(params):

    env_name = f"LunarLander-v2"
    for key, value in params.items():
        key = key.replace("_", "-")
        if value < 0:
            value = f"({get_float_to_str(abs(value))})"
        else:
            value = get_float_to_str(value)
        env_name += f"---{key}--{value}"
    return env_name


def get_param_from_env_name(env_name):
    # reverse of get_env_name_from_params
    env_parts = env_name.split("---")
    env_parts = env_parts[1:]  # Skip base environment name
    params = {}
    for env_part in env_parts:
        key, value = env_part.split("--")
        key = key.replace("-", "_")
        value.replace("-", ".")
        if value[0] == "(":
            value = value[1:-1]
            value = get_str_to_float(value) * -1
        else:
            value = get_str_to_float(value)
        params[key] = value

    return params


def use_model(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()


def get_models():
    models = [
        A2C, DDPG, PPO, SAC, TD3
    ]
    return models

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
