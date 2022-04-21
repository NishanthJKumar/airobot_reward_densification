# airobot_reward_densification
A repository with a simple robotic domain in which to test out ideas for automatic reward densification.
**NOTE:** To make full use of this library, please [install the Fast-Downward planning system](https://www.fast-downward.org/ObtainingAndRunningFastDownward) in addition to the dependencies specified in the `environment.yml` file.

# Usage/Getting Started
## Workflow

### Creating a new environment
This repository is organized around different environments which have particular, goal-based tasks and corresponding PDDL files that express the dynamics of the environment. the `envs/` folder under the main repository contains a separate folder (naming convention `<env_name>_env`) for each different environment. Within every such folder, there must be:

1. a Python file (e.g. `envs/reaching_env/reaching_task.py`) that contains an environment that must be a subclass of an [OpenAI gym environment](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e). This file must define the environment and all its necessary methods as well as register the environment. 
2. a sub-folder containing:
    1. domain and problem PDDL files (e.g. `envs/reaching_env/multiple_subgoals/`) (NOTE: in the future, these problem files will be automatically generated) that represent the dynamics of this particular environment.
    2. a python file (e.g. `envs/reaching_env/multiple_subgoals/multiple_subgoals.py`) containing a definition of a subclass that inherits from `envs.base_classifiers_class. BaseClassifiers`. This must define functions corresponding to each of the predicates in the domain (the functions must have the same name as the predicates) and expose a method called `get_typed_predicates()` that returns a dictionary whose keys are `"0-arity", "1-arity", "2-arity"` and so on until the max. arity predicates. The values must be a list of tuples where the first element is the predicate classifier function, and the following elements are the names of the types (as defined in the particular pddl domain file associated with this python file) that the arguments of this predicate function take.

### Creating a new reward densification approach
TODO (njk): Describe clearly; for now the interested reader cna refer to `shaped_reward_episodic_runner.py`.

Once these are defined, then simply set the desired settings in `main.py` and call it to either train or eval your approach with reward densification!