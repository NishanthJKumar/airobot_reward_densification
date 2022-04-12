import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
from gym.wrappers.monitor import Monitor
import fetch_block_construction
from gym.wrappers.monitoring import video_recorder
import copy
from utils import Predicates, apply_grounded_operator, get_state_grounded_atoms, apply_grounded_plan
from reaching_task import URRobotGym
from gym.envs.registration import registry, register

NUM_BLOCKS = 2

module_name = __name__

env_name = 'URReacher-v1'
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f'{module_name}:URRobotGym',
)

env_name = 'URReacher-v1'
env = gym.make(env_name)
# env = Monitor(env, directory="videos", force=True, video_callable=lambda x: x)
# vid = video_recorder.VideoRecorder(env,path="./videos/vid.mp4")
env.env._max_episode_steps = 50

obs = env.reset()
env.env.stack_only = True

def print_predicates(env):
    predicates = Predicates().get_predicates()
    objects = ['claw', 'subgoal', 'goal']
    for predicate in predicates["0-arity"]:
        print(predicate.__name__, predicate(env))

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj in ['subgoal', 'goal']:
                print(predicate.__name__, [obj], predicate(env, obj))
    
    for predicate in predicates["2-arity"]:
        for obj1 in ['claw']:
            for obj2 in ['subgoal', 'goal']:
                print(predicate.__name__, [obj1, obj2], predicate(env, obj1, obj2))  

#env.render()
print_predicates(env.env)
state_grounded_atoms = get_state_grounded_atoms(env.env)
print("State:", state_grounded_atoms)
op_name = 'move-to-subgoal'
params = ['claw', 'subgoal']
print("Apply", op_name, params)
next_grounded_state_atoms = apply_grounded_operator(state_grounded_atoms, op_name, params)
print("Next State:", next_grounded_state_atoms)
with open('sas_plan.1') as f:
    plan = [eval(line.replace('\n','').replace(' ','\', \'').replace('(','(\'').replace(')','\')')) for line in f.readlines() if 'unit cost' not in line]
print()
print("Plan:", plan)
state_grounded_atoms = get_state_grounded_atoms(env.env)
plan_grounded_atoms = apply_grounded_plan(state_grounded_atoms, plan)
for i, step in enumerate(plan_grounded_atoms):
    if i != 0:
        print(plan[i-1])
    print("State Sequence:", step)
