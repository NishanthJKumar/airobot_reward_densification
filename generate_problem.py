import numpy as np
from reaching_task import URRobotGym
from easyrl.utils.gym_util import make_vec_env
from utils import Predicates, print_predicates

module_name = __name__

env_name = "URReacher-v1"
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f"{module_name}:URRobotGym",
)

env_kwargs = (
    dict(
        with_obstacle=with_obstacle,
    )
    if not push_exp
    else dict()
)
env = make_vec_env(env_name, 1, seed=0, env_kwargs=env_kwargs)
obs = env.reset()

print_predicates(env)

# predicates = Predicates().get_predicates()
# objects = [f"object{i}" for i in range(NUM_BLOCKS)]
# problem = "(define (problem task)\n\t(:domain blocksworld)\n\t(:objects "

# for i in range(NUM_BLOCKS):
#     problem += objects[i]
#     if (i != NUM_BLOCKS - 1):
#         problem += " "

# problem += ")\n\t(:init\n\t\t"

# for predicate in predicates["0-arity"]:
#     if (predicate(env)):
#         problem += "(" + predicate.__name__ + ")\n\t\t"

# for predicate in predicates["1-arity"]:
#     for obj in objects:
#         if (predicate(env, obj)):
#             problem += "(" + predicate.__name__ + " " + obj + ")\n\t\t"

# for predicate in predicates["2-arity"]:
#     for obj1 in objects:
#         for obj2 in objects:
#             if (obj1 != obj2 and predicate(env, obj1, obj2)):
#                 problem += "(" + predicate.__name__ + " " + obj1 + " " + obj2 + ")\n\t\t"

# problem += "\n\t)\n\t(:goal (and \n\t\t(on object1 object0))\n\t)\n)"

# with open("problem.pddl", "w") as f:
#     f.write(problem)
