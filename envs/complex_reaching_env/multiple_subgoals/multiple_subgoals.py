import gym

from envs.base_classifiers_class import BaseClassifiers
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
import os

class ReachingMultipleSubgoalsComplexClassfiers(BaseClassifiers):

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

    def get_path_to_domain_and_problem_files(self):
        return (os.path.abspath("envs/complex_reaching_env/multiple_subgoals/goal-multiple-subgoal-domain.pddl"), os.path.abspath("envs/complex_reaching_env/multiple_subgoals/goal-multiple-subgoal-problem.pddl"))

