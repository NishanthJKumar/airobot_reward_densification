
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
NUM_BLOCKS = 2
max_plan_step_reached = 0

class Predicates:

    def at(self, env, gripper, loc):
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        elif self.is_subgoal1(env, loc): 
            for pos in env._subgoal1_pos:
                if np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - pos[:2]) < env._dist_threshold:
                    return True
            return False
        elif self.is_subgoal2(env, loc):
            for pos in env._subgoal2_pos:
                if np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - pos[:2]) < env._dist_threshold:
                    return True
            return False
        elif self.is_subgoal3(env, loc):
            for pos in env._subgoal3_pos:
                if np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - pos[:2]) < env._dist_threshold:
                    return True
            return False
        else:
            raise ValueError(f"loc should be either 'goal' or 'subgoal' not '{loc}'")

    def is_goal(self, env, loc):
        return loc == "goal"
    
    def is_subgoal1(self, env, loc):
        return loc == "subgoal1"

    def is_subgoal2(self, env, loc):
        return loc == "subgoal2"

    def is_subgoal3(self, env, loc):
        return loc == "subgoal3"

    def get_typed_predicates(self):
        # TODO can this be read from domain file?
        # This needs to specify typed predicates
        return {"0-arity": [], "1-arity": [(self.is_goal, "location"), (self.is_subgoal1, "location"), (self.is_subgoal2, "location"), (self.is_subgoal3, "location")], "2-arity": [(self.at, "gripper", "location")]}


class GroundingUtils:

    def __init__(self, domain_file_name, problem_file_name):
        self.domprob = pddlpy.DomainProblem(domain_file_name, problem_file_name)

    def get_state_grounded_atoms(self, env):
        state_grounded_atoms = []

        predicates = Predicates().get_typed_predicates()
        # TODO (wmcclinton) get objects with types from pddl
        objects = [(obj, obj_type) for obj, obj_type in self.domprob.problem.objects.items()]
        for predicate in predicates["0-arity"]:
            state_grounded_atoms.append([(predicate[0].__name__,), predicate[0](env)])

        for predicate in predicates["1-arity"]:
            for obj in objects:
                if obj[1] == predicate[1]:
                    state_grounded_atoms.append([(predicate[0].__name__, obj[0]), predicate[0](env, obj[0])])
        
        for predicate in predicates["2-arity"]:
            for obj1 in objects:
                for obj2 in objects:
                    if obj1[1] == predicate[1] and obj2[1] == predicate[2]:
                        state_grounded_atoms.append([(predicate[0].__name__, obj1[0], obj2[0]), predicate[0](env, obj1[0], obj2[0])]) 

        return [atom[0] for atom in state_grounded_atoms if atom[1]]

    def apply_grounded_operator(self, state_grounded_atoms, op_name, params):
        for o in self.domprob.ground_operator(op_name):
            if params == list(o.variable_list.values()) and o.precondition_pos.issubset(state_grounded_atoms):
                next_state_grounded_atoms = copy.deepcopy(state_grounded_atoms)
                for effect in o.effect_pos:
                    next_state_grounded_atoms.append(effect)
                for effect in o.effect_neg:
                    next_state_grounded_atoms.remove(effect)
                return next_state_grounded_atoms
        return None

    def apply_grounded_plan(self, state_grounded_atoms, plan):
        plan_grounded_atoms = []
        plan_grounded_atoms.append(state_grounded_atoms)
        for ground_operator in plan:
            op_name = ground_operator[0]
            params = list([ground_operator[1]] if len(ground_operator[1]) == 2 else ground_operator[1:])
            plan_grounded_atoms.append(self.apply_grounded_operator(plan_grounded_atoms[-1], op_name, params))
        return plan_grounded_atoms

    def phi(self, state_grounded_atoms, plan):
            for i, grounded_atoms in enumerate(plan[max_plan_step_reached:]):
                if grounded_atoms == state_grounded_atoms:
                    return i + max_plan_step_reached
            return max_plan_step_reached

    def get_shaped_reward(self, env, state, previous_state_grounded_atoms, next_state_grounded_atoms, plan):
        global max_plan_step_reached
        dist_to_goal = np.linalg.norm(state - env._goal_pos[:2])
        success = dist_to_goal < env._dist_threshold
        reward = 1 if success else 0

        prev_phi = self.phi(previous_state_grounded_atoms, plan)
        if max_plan_step_reached < prev_phi:
            max_plan_step_reached = prev_phi

        f = self.phi(next_state_grounded_atoms, plan) - self.phi(previous_state_grounded_atoms, plan)
        reward = reward + f
        # reward = -dist_to_goal
        info = dict(success=success)
        return reward, info
