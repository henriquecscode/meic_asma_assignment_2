from training_utils import *
from utils import *
from modified_env_utils import *
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3.common.env_checker import check_env


import sys
start_env = 0
models = get_models()
is_verbose = False
EPISODES = 15
TIMESTEPS = 10000

if len(sys.argv) > 1:
    start_env = int(sys.argv[1])
    if len(sys.argv) > 2:
        if sys.argv[2] == "all":
            models = get_models()
        else:
            idx = int(sys.argv[2])
            models = [models[idx]]


def learn_all_models(verbose=False):
    for idx, (env, env_name) in enumerate(gen_envs()):
        if idx < start_env:
            continue
        for model_cls in models:
            learn_model_env(model_cls, env, env_name, verbose)


def learn_model_env(model_cls, env, env_name, verbose=False):
    check_env(env)
    model = model_cls("MlpPolicy", env, verbose=verbose,
                      tensorboard_log=LOGS_DIR)
    path = get_model_path(model, env_name, "simple")
    learning_routine = create_training_routine(model, path)
    print(f"Starting training {model_cls.__name__} on {env_name}")
    learning_routine(EPISODES, TIMESTEPS)
    print(f"Finished training {model_cls.__name__} on {env_name}")


def learn_all_models_threaded():
    pool = ThreadPoolExecutor(max_workers=len(models))
    for idx, envs in enumerate(gen_multiple_envs(len(models))):
        if idx < start_env:
            continue
        futures = []
        for i in range(len(models)):
            env, env_name = envs[i]
            model = models[i]
            future = pool.submit(learn_model_env, model, env, env_name)
            futures.append(future)
        _ = [future.result() for future in futures]
        print("Finished training all models on ", env_name)


if __name__ == '__main__':
    verbose = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--verbose":
                verbose = True
    set_modified_parameters_functions(get_single_variable_parameters)
    learn_all_models(verbose)
print("Finished training all models")
