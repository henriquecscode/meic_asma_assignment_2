# [meic_asma_assignment_2](https://github.com/henriquecscode/meic_asma_assignment_2)

# Introduction

This project train stable baseline's model on the Lunar Lander environment. Multiple variations of the environment are used.
## Installation and setup

This project runs with stable baselines and gym.
The versions for each package MUST be:

* stable_baselines==1.8.0
* gym==0.21.0

## Running

For running refer to the section [Training and using models](#training-and-using-models)

## Version compatibility

In order to use features from the later versions of the Lunar Lander environment, the source code from version 0.24.0 was used. Adaptations had to be done in order to make the code compatible with the older version of gym.

## Files

### Utilitary 
#### Utils

Contains utilitary function
The models present in the project can be changed in the start of the file in `MODELS` variable. Make sure they are compatible with a Box action space.

#### Training utils

Implements the learning routine that is used to train the models. Advised not to change

#### Modified env

Contains the environment where the algorithms are ran. If one wishes to add or change behavior of the environment, this is the place to do it. Use of wrappers is advised. Proceed with caution. 

To get an environment, the easiest way is to call `get_wrapped_lunar_environment` with the unpacked parameters dictionary. For example `get_wrapped_lunar_environment(**params)`. 

#### Modified env utils

Contains utilitary functions that are used for the management of the modified environments.
To get the standard environments you can use the functions
* `gen_envs` - generator with the standard environments
* `get_envs` - list with the standard environments
* `gen_multiple_envs` - Takes a integer n. Generator that, for each iteration, returns a list with n equal environments. Intended for parallelization.

There are two standard sets of environments.

* Combinatoric variable modification - Changes all parameters of the environment at the same time
  * Default option
  * `get_all_permutated_parameters()` can be used to get the parameters list manually
* One variable modification - Only changes one parameter of the environment at the same time
  * `get_single_variable_parameters()` can be used to get the parameters list manually

The function `set_modified_parameters_functions` can be called with any of the previous functions to change the standard environments that are retrieved in the env getters.

Standard environment parameters are equivalent to
```
standard_params = {
    "gravity": -10.0,
    "max_thrust": 1.0,
    "wind_power": 0.0,
    "turbulence_power": 0.0,
    "failure_rate": 0.0,
    "byzantine_rate": 0.0
}
```

To convert from parameters to an environment name, use `get_env_name_from_params`. To convert from an environment name to parameters, use `get_params_from_env_name`.

### Training and using models
 Whenever a model is training, press Ctrl^C will notify the program to terminate the training as soon as possible. Once the episode finishes the model will be saved and the program will terminate as soon as possible. If there were multiple trainings scheduled, for example for different environments and models, these will show up but no changes will have occurred.

#### Learn all models

Learn all models will train the models in the `MODELS` variable in `utils.py` on the environments in `gen_envs()`. As of now, this corresponds to the *One variable modification* modified environments. The models will be trained for 15 episodes of 10k iterations each, by default. The models will be saved in the `models/` directory.

The following command lines are accepted. Take into account order of arguments as these don't have flags.
* `<startingEnvironment>` - First argument. 
  * If present, defines the first environment to train the models on. If not present, the first environment will be the first in the list of environments.
* `<model>` - Second argument. 
  * If present, defines the model to be trained. If not present, all models will be trained.

* `--verbose` 
  * If present, the model will be initialized as verbose and information on training will be displayed in real time.

#### Retrain model

Retrains a model.
* If no command line arguments are passed, a new model is created and trained on the default environment with a PPO algorithm.
* To actually retrain a specific model, pass as the first argument a model name and/or an iterations file. If no iterations file is provided the latest will be used. Do not use the `models/` directory path in the model name.

You can also specify how you want the training to proceed. By default, the training will be for 15 episodes of 10k iterations. If you wish to alter this behavior you can pass the following arguments:

* `--episodes=<numberOfEpisodes>` - Number of episodes to train the model for
* `--iterations=<numberOfIterations>` - Number of iterations to train the model for
* `--totalIterations=<numberOfTotalIterations`> - Total number of iterations to train the model for. Overrides number of episodes. Number of iterations can still be passed, otherwise Default. Number of episodes is deduced based on the two.

#### Use model gym

To use a model and see it operate a lander in real time.
To change the model you can:
* Change the DEFAULT_MODEL inside the file
* Pass it a command line argument with the model path. The model path must include the zip file itself. No need for the `models/` directory path. For example `python use_model_gym.py 2023-05-21-13-14-05_LunarLander-v2---wind-power--15-0_SAC_simple/150000.zip`

#### Test Env

To test some features of the modified environments, a manual version of the environment was developed. 
The commands to operate the lander are

* `Key Up` - Main engine
* `Key Left` - Move left (right engine)
* `Key Right` - Move right (left engine)