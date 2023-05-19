import datetime
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C, PPO
import os
MODELS_DIR = 'models/'
LOGS_DIR = 'logs/'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


def get_date():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def get_model_path(model: BaseAlgorithm, env_name, suffix=""):
    return f"{get_date()}_{env_name}_{get_model_type_name(model)}_{suffix}"


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
    elif model_type == "PPO":
        cls = PPO

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
