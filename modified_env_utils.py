from modified_env import get_wrapped_lunar_environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np

ITERATIONS = 20000
max_thrust = 0.75
failure_rate = 0.5
byzantine_rate = 0.5

max_thurst_min = 0.75
max_thrust_max = 1.0
max_thrust_inc = 0.25
max_thrust_num = int(abs(max_thrust_max - max_thurst_min) / max_thrust_inc + 1)

failure_rate_min = 0.0
failure_rate_max = 0.5
failure_rate_inc = 0.25
failure_rate_num = int(
    abs(failure_rate_max - failure_rate_min) / failure_rate_inc + 1)

byzantine_rate_min = 0.0
byzantine_rate_max = 0.5
byzantine_rate_inc = 0.25
byzantine_rate_num = int(
    abs(byzantine_rate_max - byzantine_rate_min) / byzantine_rate_inc + 1)

wind_power_min = 0.0
wind_power_max = 15.0
wind_power_inc = 5.0
wind_power_num = int(abs(wind_power_max - wind_power_min) / wind_power_inc + 1)

turbulence_power_min = 0.0
turbulence_power_max = 0.0
turbulence_power_inc = 1.0
turbulence_power_num = int(
    abs(turbulence_power_max - turbulence_power_min) / turbulence_power_inc + 1)

gravity_min = -10.0
gravity_max = -10.0
gravity_inc = 1.0
gravity_num = int(abs(gravity_max - gravity_min) / gravity_inc + 1)

standard_params = {
    "gravity": -10.0,
    "max_thrust": 1.0,
    "wind_power": 0.0,
    "turbulence_power": 0.0,
    "failure_rate": 0.0,
    "byzantine_rate": 0.0
}

def _max_thrust_modification(max_thrust):
    env = get_wrapped_lunar_environment(continuous=True, max_thrust=max_thrust)
    env_name = f"LunarLander-v2-modified-thrust-{max_thrust}"
    return env, env_name


def _failure_rate_modification(failure_rate):
    env = get_wrapped_lunar_environment(
        continuous=True, failure_rate=failure_rate)
    env_name = f"LunarLander-v2-modified-failure-{failure_rate}"
    return env, env_name


def _byzantine_rate_modification(byzantine_rate):
    env = get_wrapped_lunar_environment(
        continuous=True, byzantine_rate=byzantine_rate)
    env_name = f"LunarLander-v2-modified-byzantine-{byzantine_rate}"
    return env, env_name


def _both_errors_modification(failure_rate, byzantine_rate):
    env = get_wrapped_lunar_environment(
        continuous=True, failure_rate=failure_rate, byzantine_rate=byzantine_rate)
    env_name = f"LunarLander-v2-modified-failure-{failure_rate}-byzantine-{byzantine_rate}"
    return env, env_name


def parameters_is_standard(params):
    for key in params:
        if params[key] != standard_params[key]:
            return False
    return True

def get_single_variable_parameters():

    all_params = []
    for gravity in np.linspace(gravity_min, gravity_max, gravity_num):
        params = {
            "gravity": gravity,
        }
        all_params.append(params)
    for max_thrust in np.linspace(max_thrust_max, max_thurst_min, max_thrust_num):
        params = {
            "max_thrust": max_thrust,
        }
        all_params.append(params)

    for wind_power in np.linspace(wind_power_min, wind_power_max, wind_power_num):
        params = {
            "wind_power": wind_power,
        }
        all_params.append(params)

    for turbulence_power in np.linspace(turbulence_power_min, turbulence_power_max, turbulence_power_num):
        params = {
            "turbulence_power": turbulence_power,
        }
        all_params.append(params)

    for failure_rate in np.linspace(failure_rate_min, failure_rate_max, failure_rate_num):
        params = {
            "failure_rate": failure_rate,
        }
        all_params.append(params)

    for byzantine_rate in np.linspace(byzantine_rate_min, byzantine_rate_max, byzantine_rate_num):
        params = {
            "byzantine_rate": byzantine_rate,
        }
        all_params.append(params)


    filtered_all_params = list(filter(lambda x: not parameters_is_standard(x), all_params))
    return [{}] + filtered_all_params


def get_all_permutated_parameters():
    failure_params = []
    byzantine_params = []

    for gravity in np.linspace(gravity_min, gravity_max, gravity_num):
        for max_thrust in np.linspace(max_thrust_max, max_thurst_min, max_thrust_num):
            for wind_power in np.linspace(wind_power_min, wind_power_max, wind_power_num):
                for turbulence_power in np.linspace(turbulence_power_min, turbulence_power_max, turbulence_power_num):
                    params = {
                        "gravity": gravity,
                        "max_thrust": max_thrust,
                        "wind_power": wind_power,
                        "turbulence_power": turbulence_power
                    }
                    for failure_rate in np.linspace(failure_rate_min, failure_rate_max, failure_rate_num):
                        failure_dict = dict(params)
                        errors_param = {
                            "failure_rate": failure_rate,
                            "byzantine_rate": 0.0
                        }
                        failure_dict.update(errors_param)
                        failure_params.append(failure_dict)
                    # for byzantine_rate in np.linspace(byzantine_rate_min, byzantine_rate_max, byzantine_rate_num):
                    #     byzantine_dict = dict(params)
                    #     errors_param = {
                    #         "failure_rate": 0.0,
                    #         "byzantine_rate": byzantine_rate
                    #     }
                    #     byzantine_dict.update(errors_param)
                    #     byzantine_params.append(byzantine_dict)

    failure_params.remove(standard_params)
    # byzantine_params.remove(standard_params)
    params = [{}] + failure_params + byzantine_params
    return params

def get_standard_environment_parameters():
    return [{}]

def get_failure_rates_params():
    return [
        {'failure_rate': 0.05
        },
        {'failure_rate': 0.1}
    ]

modified_parameters_functions = get_all_permutated_parameters
def set_modified_parameters_functions(func):
    global modified_parameters_functions
    modified_parameters_functions = func

def get_modified_parameters():
    return modified_parameters_functions()

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


def get_envs():
    return [env for env in gen_envs()]

def gen_envs():
    params = get_modified_parameters()
    for params in params:
        env = get_wrapped_lunar_environment(**params)
        env_name = get_env_name_from_params(params)
        yield (env, env_name)

def gen_multiple_envs(num_envs):
    params = get_modified_parameters()
    for params in params:
        envs = []
        for _ in range(num_envs):
            env = get_wrapped_lunar_environment(**params)
            env_name = get_env_name_from_params(params)
            envs.append((env, env_name))
        yield envs

if __name__ == '__main__':
    # env, env_name = failure_rate_modification(failure_rate)
    # env, env_name = both_errors_modification(failure_rate, byzantine_rate)
    # model = train_modified_env(env, env_name)
    # use_model(model, env)
    set_modified_parameters_functions(get_single_variable_parameters)
    params = get_modified_parameters()
    print(f"Number of environments: {len(params)}")
    consistent = True
    for params in params:
        # env = get_wrapped_lunar_environment(**params)
        env_name = get_env_name_from_params(params)
        env = get_wrapped_lunar_environment(**params)
        new_params = get_param_from_env_name(env_name)
        consistent = consistent and new_params == params
        if new_params != params:
            print(f"Env name is {env_name}")
            print(f"Params are {params}")
            print(f"New params are {new_params}")
        print(f"Consistent env name {env_name} is {new_params == params}")
    # pass
    if consistent:
        print("modified_env_utils working")
    else: 
        print("modified_env_utils is not consistent")
