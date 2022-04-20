import argparse
from reaching_task import URRobotGym
from easyrl.utils.gym_util import make_vec_env
from utils import Predicates
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

predicates = Predicates().get_predicates()
for predicate in predicates["2-arity"]:
    for loc in range(2 ** env.envs[0]._granularity):
        print(predicate.__name__, ["claw", "loc" + str(loc)], predicate(env.envs[0], "claw", "loc" + str(loc)))
