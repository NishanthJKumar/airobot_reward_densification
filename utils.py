
import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
from gym.wrappers.monitor import Monitor
from gym.wrappers.monitoring import video_recorder
import copy
import pddlpy

domprob = pddlpy.DomainProblem('goal-subgoal-domain.pddl', 'goal-subgoal-problem.pddl')
NUM_BLOCKS = 2

class Predicates:

    # TODO: add necessary predicate classifiers
    def at(self, env, gripper, loc):
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        else: 
            if loc == "subgoal":
                for pos in env._subgoal_pos:
                    if np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - pos[:2]) < env._dist_threshold:
                        return True
                return False
            else:
                raise ValueError(f"loc should be either 'goal' or 'subgoal' not '{loc}'")

    def is_goal(self, env, loc):
        return loc == "goal"

    def get_predicates(self):
        return {"0-arity": [], "1-arity": [self.is_goal], "2-arity": [self.at]}


def get_state_grounded_atoms(env):
    state_grounded_atoms = []

    predicates = Predicates().get_predicates()
    objects = ['claw', 'subgoal', 'goal']
    for predicate in predicates["0-arity"]:
        state_grounded_atoms.append([(predicate.__name__,), predicate(env)])

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj in ['subgoal', 'goal']:
                state_grounded_atoms.append([(predicate.__name__, obj), predicate(env, obj)])
    
    for predicate in predicates["2-arity"]:
        for obj1 in ['claw']:
            for obj2 in  ['subgoal', 'goal']:
                state_grounded_atoms.append([(predicate.__name__, obj1, obj2), predicate(env, obj1, obj2)]) 

    return [atom[0] for atom in state_grounded_atoms if atom[1]]

def apply_grounded_operator(state_grounded_atoms, op_name, params):
    for o in domprob.ground_operator(op_name):
        if params == list(o.variable_list.values()) and o.precondition_pos.issubset(state_grounded_atoms):
            next_state_grounded_atoms = copy.deepcopy(state_grounded_atoms)
            for effect in o.effect_pos:
                next_state_grounded_atoms.append(effect)
            for effect in o.effect_neg:
                next_state_grounded_atoms.remove(effect)
            return next_state_grounded_atoms
    return None

def apply_grounded_plan(state_grounded_atoms, plan):
    plan_grounded_atoms = []
    plan_grounded_atoms.append(state_grounded_atoms)
    for ground_operator in plan:
        op_name = ground_operator[0]
        params = list([ground_operator[1]] if len(ground_operator[1]) == 2 else ground_operator[1:])
        plan_grounded_atoms.append(apply_grounded_operator(plan_grounded_atoms[-1], op_name, params))
    return plan_grounded_atoms



