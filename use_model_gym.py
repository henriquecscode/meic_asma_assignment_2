
from utils import *
from modified_env_utils import get_param_from_env_name
import sys

DEFAULT_MODEL = "2023-05-21-13-14-05_LunarLander-v2---wind-power--15-0_SAC_simple/150000.zip"
if __name__ == '__main__':
    if len(sys.argv) == 1:
        filename = DEFAULT_MODEL
    else:
        filename = sys.argv[1]

    env_name = get_env(filename)
    env_params = get_param_from_env_name(env_name)
    env = get_wrapped_lunar_environment(**env_params)
    model = load_model(filename)
    while True:
        use_model(model, env)