import argparse
import numpy as np
import envs.reaching_env.reaching_task
from easyrl.utils.gym_util import make_vec_env
from envs.reaching_env.grid_based.grid_based import ReachingGridBasedClassifiers
from envs.reaching_env.grid_based.grid_based_complex import ReachingGridBasedComplexClassifiers
from gym.envs.registration import registry, register

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--granularity', type=int, default=3, help='Number of divisions to segment the working space of the arm. Total divisions is equal to 2^{input}.')
args = parser.parse_args()

if (args.granularity <= 0):
    raise argparse.ArgumentTypeError("Granularity must be greater than 0.")

env_kwargs = (dict(with_obstacle=True, granularity=args.granularity, reward_type="pddl", pddl_type="sparse_handcrafted"))
env = make_vec_env("URReacher-v2", 1, seed=0, env_kwargs=env_kwargs)
obs = env.reset()

# Set up the header of the problem file.
problem = "(define (problem task)\n\t(:domain reaching-grid)\n\t(:objects claw - gripper "
for loc in range(2 ** env.envs[0]._granularity):
    problem += "loc" + str(loc) + " "
problem += "goal - location)\n\t(:init\n\t\t"

# Set up the initial state of the problem file.
predicates = ReachingGridBasedComplexClassifiers().get_typed_predicates()
for predicate in predicates["0-arity"]:
    if (predicate[0](env.envs[0])):
        problem += "(" + predicate[0].__name__ + ")\n\t\t"
for predicate in predicates["1-arity"]:
    if (predicate[0](env.envs[0], "goal")):
        problem += "(" + predicate[0].__name__ + " goal)\n\t\t"
for predicate in predicates["2-arity"]:
    if (predicate[0].__name__ == "at"):
        for loc in range(2 ** env.envs[0]._granularity):
            if (predicate[0](env.envs[0], "claw", "loc" + str(loc))):
                problem += "(" + predicate[0].__name__ + " claw loc" + str(loc) + ")\n\t\t"
    elif (predicate[0].__name__ == "neighbors"):
        for loc1 in range(2 ** env.envs[0]._granularity):
            for loc2 in range(2 ** env.envs[0]._granularity):
                if (loc1 != loc2 and predicate[0](env.envs[0], "loc" + str(loc1), "loc" + str(loc2))):
                    problem += "(" + predicate[0].__name__ + " loc" + str(loc1) + " loc" + str(loc2) + ")\n\t\t"
        for loc in range(2 ** env.envs[0]._granularity):
            if (predicate[0](env.envs[0], "goal", "loc" + str(loc))):
                problem += "(" + predicate[0].__name__ + " goal loc" + str(loc) + ")\n\t\t"
            if (predicate[0](env.envs[0], "loc" + str(loc), "goal")):
                problem += "(" + predicate[0].__name__ + " loc" + str(loc) + " goal)\n\t\t"

problem += "\n\t)\n\t(:goal (and \n\t\t(at claw goal))\n\t)\n)"

with open("reaching-grid-complex-problem" + str(env.envs[0]._granularity) + ".pddl", "w") as f:
    f.write(problem)