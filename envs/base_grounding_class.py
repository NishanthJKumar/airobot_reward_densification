from typing import List
import pddlpy
import gym
import copy
from base_classifiers_class import BaseClassifiers

class BaseGrounding:

    def __init__(self, domain_file_path: str, problem_file_path: str, vec_env: List[gym.env], classifiers: BaseClassifiers):
        self.domain_file_path = domain_file_path
        # TODO: make problem file automatically instead of taking in
        # right now. This can be done by just running the classifiers
        # on the vec_env and generating the initial state!
        self.problem_file_path = problem_file_path
        self.vec_env = vec_env
        self.classifiers = classifiers

        # NOTE: In the future, we should be generaitng this problem_file_path
        # within this init method.
        self.domprob = pddlpy.DomainProblem(domain_file_path, problem_file_path)

    def get_state_grounded_atoms(self, env):
        state_grounded_atoms = []

        predicates = self.classifiers.get_typed_predicates()
        # TODO (wmcclinton) get objects with types from pddl
        objects = [(obj, obj_type) for obj, obj_type in self.domprob.problem.objects.items()]
        
        # TODO: make this cleaner so that we can automatically get things up to
        # whatever arity defined in the file.
        # TODO: protect against these keys not existing.
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
