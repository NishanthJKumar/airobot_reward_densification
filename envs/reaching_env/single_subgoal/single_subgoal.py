import gym

from envs.base_classifiers_class import BaseClassifiers
gym.logger.set_level(gym.logger.DEBUG)
import numpy as np
import os

class SingleSubgoalClassfiers(BaseClassifiers):

    def at(self, env, gripper, loc):
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        elif loc == "subgoal":
            for subgoal_pos in env._subgoal2_pos: 
                if np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - subgoal_pos[:2]) < env._dist_threshold:
                    return True
            return False
        else:
            raise ValueError(f"loc should be either 'goal' or 'subgoal' not '{loc}'")

    def is_goal(self, env, loc):
        return loc == "goal"

    def get_typed_predicates(self):
        # TODO can this be read from domain file?
        # This needs to specify typed predicates
        return {"0-arity": [], "1-arity": [(self.is_goal, "location")], "2-arity": [(self.at, "gripper", "location")]}

    def get_path_to_domain_and_problem_files(self):
        return (os.path.abspath("envs/reaching_env/single_subgoal/goal-subgoal-domain.pddl"), os.path.abspath("envs/reaching_env/single_subgoal/goal-subgoal-problem.pddl"))
