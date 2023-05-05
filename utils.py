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
    if len(filename_parts) < 6:
        return ""
    env_name = filename_parts[6]
    return env_name
