import numpy as np
import gym
from envs.lunar_lander import LunarLander

LUNAR_NAME = "LunarLander-v2"
LUNAR_CONTINOUS_NAME = "LunarLanderContinuous-v2"
ANT_NAME = "Ant-v3"

def get_lunar_env(continuous: bool = False,
                  gravity: float = -10.0,
                  enable_wind: bool = False,
                  wind_power: float = 15.0,
                  turbulence_power: float = 1.5,
                  ):
    env = LunarLander(
        continuous=continuous,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power)
    # env = gym.make(LUNAR_NAME,
    #                continuous=continuous,
    #                gravity=gravity,
    #                enable_wind=enable_wind,
    #                wind_power=wind_power,
    #                turbulence_power=turbulence_power)
    # if not continuous:
    #     env = gym.make(LUNAR_NAME)
    # else:
    #     env = gym.make(LUNAR_CONTINOUS_NAME,
    #                    gravity=gravity,
    #                    enable_wind=enable_wind,
    #                    wind_power=wind_power,
    #                    turbulence_power=turbulence_power)
    return env


class LimitThrustWrapper(gym.ActionWrapper):
    def __init__(self, env, max_thrust=0.75):
        super().__init__(env)
        self.max_thrust = max_thrust
        self.action_space = gym.spaces.Box(
            low=-max_thrust, high=max_thrust, shape=(2,), dtype=np.float32)

    def action(self, action):
        return np.clip(action, -self.max_thrust, self.max_thrust)

class RandomThrustFailureWrapper(gym.ActionWrapper):
    def __init__(self, env, failure_rate=0.1):
        super().__init__(env)
        self.failure_rate = failure_rate
        self.action_space = env.action_space

    def action(self, action ):

        if np.random.random() < self.failure_rate:
            #fill action with 0's
            action = np.zeros_like(action)
        return action

class ByzantineThrusterWrapper(gym.ActionWrapper):
    def __init__(self, env, byzantine_rate=0.1):
        super().__init__(env)
        self.byzantine = byzantine_rate
        self.action_space = env.action_space

    def action(self, action ):
        if np.random.random() < self.byzantine:
            # random action
            action = np.random.uniform(-1, 1, size=action.shape)
        return action

def get_wrapped_lunar_environment(
                            gravity: float = -10.0,
                            wind_power: float = 0.0,
                            turbulence_power: float = 0.0,
                            max_thrust: float = None,
                            failure_rate: float = None,
                            byzantine_rate: float = None,
                            ):
    enable_wind = wind_power != 0.0 or turbulence_power != 0.0
    env = get_lunar_env(True, gravity, enable_wind,
                        wind_power, turbulence_power)
    if max_thrust is not None and max_thrust != 1.0:
        env = LimitThrustWrapper(env, max_thrust)
    if failure_rate is not None and byzantine_rate is not None and failure_rate != 0.0 and byzantine_rate != 0.0:
        raise ValueError("Cannot have both failure rate and byzantine rate")
    if failure_rate is not None and failure_rate != 0.0:
        env = RandomThrustFailureWrapper(env, failure_rate)
    if byzantine_rate is not None and byzantine_rate != 0.0:
        env = ByzantineThrusterWrapper(env, byzantine_rate)
    return env


def get_environment_action_space(env=None):
    if env is None:
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    return env.action_space

def get_environment_observation_space(env=None):
    if env is None:
        return gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
    return env.observation_space


if __name__ == '__main__':
    env = get_wrapped_lunar_environment(failure_rate=0.1)
    print(env.action_space)
    print(env.observation_space)
    pass
