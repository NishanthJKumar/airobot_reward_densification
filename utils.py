
import gym
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
from gym.wrappers.monitor import Monitor
from gym.wrappers.monitoring import video_recorder
import copy
import pddlpy

# Only one Subgoal.
# domprob = pddlpy.DomainProblem('goal-subgoal-domain.pddl', 'goal-subgoal-problem.pddl')
# Multiple Subgoals.
domprob = pddlpy.DomainProblem('goal-multiple-subgoal-domain.pddl', 'goal-multiple-subgoal-problem.pddl')
NUM_BLOCKS = 2
max_plan_step_reached = 0

class Predicates:

    def at(self, env, gripper, loc):
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        elif "loc" in loc:
            loc_index = int(loc[len("loc"):])
            if env._granularity % 2 == 0:
                rows, cols = env._granularity, env._granularity
            else:
                rows, cols = env._granularity - 1, 2 ** env._granularity / (env._granularity - 1)
            loc_x, loc_y = loc_index // cols, loc_index % cols
            xmin, ymin = env._xy_bounds[:, 0]
            xmax, ymax = env._xy_bounds[:, 1]
            x_lower_bound = xmin + (xmax - xmin) / rows * loc_x
            x_upper_bound = xmin + (xmax - xmin) / rows * (loc_x + 1)
            y_lower_bound = ymin + (ymax - ymin) / cols * loc_y
            y_upper_bound = ymin + (ymax - ymin) / cols * (loc_y + 1)
            return (x_lower_bound <= env.robot.arm.get_ee_pose()[0][0] <= x_upper_bound) and (y_lower_bound <= env.robot.arm.get_ee_pose()[0][1] <= y_upper_bound)
        else:
            raise ValueError(f"loc should be either 'goal' or must start with 'loc' not '{loc}'")

    def is_goal(self, env, loc):
        return loc == "goal"
    
    def get_predicates(self):
        return {"0-arity": [], "1-arity": [self.is_goal], "2-arity": [self.at]}


def get_state_grounded_atoms(env):
    state_grounded_atoms = []

    predicates = Predicates().get_predicates()
    objects = ['claw', 'subgoal1', 'subgoal2', 'subgoal3', 'goal']
    for predicate in predicates["0-arity"]:
        state_grounded_atoms.append([(predicate.__name__,), predicate(env)])

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj in ['subgoal1', 'subgoal2', 'subgoal3', 'goal']:
                state_grounded_atoms.append([(predicate.__name__, obj), predicate(env, obj)])
    
    for predicate in predicates["2-arity"]:
        for obj1 in ['claw']:
            for obj2 in  ['subgoal1', 'subgoal2', 'subgoal3', 'goal']:
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

def phi(state_grounded_atoms, plan):
        for i, grounded_atoms in enumerate(plan[max_plan_step_reached:]):
            if grounded_atoms == state_grounded_atoms:
                return i + max_plan_step_reached
        return max_plan_step_reached

def get_shaped_reward(env, state, previous_state_grounded_atoms, next_state_grounded_atoms, plan):
    global max_plan_step_reached
    dist_to_goal = np.linalg.norm(state - env._goal_pos[:2])
    success = dist_to_goal < env._dist_threshold
    reward = 1 if success else 0

    prev_phi = phi(previous_state_grounded_atoms, plan)
    if max_plan_step_reached < prev_phi:
        max_plan_step_reached = prev_phi

    f = phi(next_state_grounded_atoms, plan) - phi(previous_state_grounded_atoms, plan)
    reward = reward + f
    # reward = -dist_to_goal
    info = dict(success=success)
    return reward, info
