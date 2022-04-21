import argparse
from reaching_task import URRobotGym
from easyrl.utils.gym_util import make_vec_env
from grid_problem_predicates import Predicates
from gym.envs.registration import registry, register

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--granularity', type=int, default=3, help='Number of divisions to segment the working space of the arm. Total divisions is equal to 2^{input}.')
args = parser.parse_args()

if (args.granularity <= 0):
    raise argparse.ArgumentTypeError("Granularity must be greater than 0.")

module_name = __name__

env_name = "URReacher-v1"
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f"{module_name}:URRobotGym",
)

env_kwargs = (dict(with_obstacle=True, granularity=args.granularity))
env = make_vec_env(env_name, 1, seed=0, env_kwargs=env_kwargs)
obs = env.reset()

def occupied(env, entity, loc):
    loc_index = int(loc[len("loc"):])
    if env._granularity % 2 == 0:
        square = int(np.sqrt(2 ** env._granularity))
        rows, cols = square, square
    else:
        square = int(np.sqrt(2 ** (env._granularity - 1)))
        rows, cols = square, int((2 ** env._granularity) / square)
    loc_x, loc_y = loc_index // cols, loc_index % cols
    xmin, ymin = env._xy_bounds[:, 0]
    xmax, ymax = env._xy_bounds[:, 1]
    x_lower_bound = xmin + (xmax - xmin) / rows * loc_x
    x_upper_bound = xmin + (xmax - xmin) / rows * (loc_x + 1)
    y_lower_bound = ymin + (ymax - ymin) / cols * loc_y
    y_upper_bound = ymin + (ymax - ymin) / cols * (loc_y + 1)

    if entity == "obstacle":
        wall_min_x, wall_min_y = 0.5 - (0.18/2), 0.15 - (0.01/2)
        wall_max_x, wall_max_y = 0.5 + (0.18/2), 0.15 + (0.01/2)
        if (x_upper_bound < wall_min_x or x_lower_bound > wall_max_x) and (y_upper_bound < wall_min_y or y_lower_bound > wall_max_y):
            return False
        else:
            return True
    elif entity == "goal":
        goal_pos_x, goal_pos_y = env._goal_pos[:2]
        if (x_lower_bound <= goal_pos_x <= x_upper_bound and y_lower_bound <= goal_pos_y <= y_upper_bound):
            return True
    else:
        raise Exception(f"Entity should be either 'obstacle' or 'goal', not {entity}")

    return False

# Set up the header of the problem file.
problem = "(define (problem task)\n\t(:domain reaching-grid)\n\t(:objects claw - gripper "
for loc in range(2 ** env.envs[0]._granularity):
    problem += "loc" + str(loc) + " "
problem += "goal - location)\n\t(:init\n\t\t"

# Set up the initial state of the problem file.
predicates = Predicates().get_predicates()
for predicate in predicates["0-arity"]:
    if (predicate(env.envs[0])):
        problem += "(" + predicate.__name__ + ")\n\t\t"
for predicate in predicates["1-arity"]:
    if (predicate(env.envs[0], "goal")):
        problem += "(" + predicate.__name__ + " goal)\n\t\t"
for predicate in predicates["2-arity"]:
    if (predicate.__name__ == "at"):
        for loc in range(2 ** env.envs[0]._granularity):
            if (predicate(env.envs[0], "claw", "loc" + str(loc))):
                problem += "(" + predicate.__name__ + " claw loc" + str(loc) + ")\n\t\t"
    elif (predicate.__name__ == "neighbors"):
        for loc1 in range(2 ** env.envs[0]._granularity):
            for loc2 in range(2 ** env.envs[0]._granularity):
                if (loc1 != loc2 and predicate(env.envs[0], "loc" + str(loc1), "loc" + str(loc2)) and (not occupied(env.envs[0], "obstacle", "loc" + str(loc)))):
                    problem += "(" + predicate.__name__ + " loc" + str(loc1) + " loc" + str(loc2) + ")\n\t\t"


for loc in range(2 ** env.envs[0]._granularity):
    if (occupied(env.envs[0], "goal", "loc" + str(loc))):
        problem += "(neighbors goal loc" + str(loc) + ")\n\t\t"
        problem += "(neighbors loc" + str(loc) + " goal)\n\t\t"

problem += "\n\t)\n\t(:goal (and \n\t\t(at claw goal))\n\t)\n)"

with open("reaching-grid-problem" + str(env.envs[0]._granularity) + ".pddl", "w") as f:
    f.write(problem)
