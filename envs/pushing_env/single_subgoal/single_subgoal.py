import os
import gym
import numpy as np
from envs.base_classifiers_class import BaseClassifiers

gym.logger.set_level(gym.logger.DEBUG)

# Only one Subgoal.

class SingleSubgoalClassfiers(BaseClassifiers):

    def robot_at(self, env, gripper, loc):
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        elif self.is_subgoal0(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._subgoal0_pos[:2]) < env._dist_threshold
        elif self.is_subgoal1(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._subgoal1_pos[:2]) < env._dist_threshold:
        else:
            raise ValueError(f"loc should be either 'goal' or 'subgoal' not '{loc}'")

    def object_at(self, env, obj, loc):
        object_pos = np.zeros(2)
        if (obj == "box"):
            object_pos, _ = self.robot.pb_client.get_body_state(self._box_id)[:2]
        else:
            raise ValueError(f"obj should be 'box1' nor '{obj}'")
        if self.is_goal(env, loc):
            return np.linalg.norm(object_pos[:2] - env._goal_pos[:2]) < env._dist_threshold
        elif self.is_subgoal0(env, loc):
            return np.linalg.norm(object_pos[:2] - env._subgoal0_pos[:2]) < env._dist_threshold
        elif self.is_subgoal1(env, loc):
            return np.linalg.norm(object_pos[:2] - env._subgoal1_pos[:2]) < env._dist_threshold:
        else:
            raise ValueError(f"loc should be either 'goal' or 'subgoal0' or 'subgoal1' not '{loc}'")

    def is_goal(self, env, loc):
        return loc == "goal"

    def is_subgoal0(self, env, loc):
        return loc == "subgoal0"

    def is_subgoal1(self, env, loc):
        return loc == "subgoal1"

    def get_typed_predicates(self):
        # TODO can this be read from domain file?
        # This needs to specify typed predicates
        return {"0-arity": [], "1-arity": [(self.is_goal, "location")], "2-arity": [(self.robot_at, "gripper", "location"), (self.object_at, "box", "location")]}

    def get_path_to_domain_and_problem_files(self):
        return (os.path.abspath("envs/pushing_env/single_subgoal/goal-subgoal-domain.pddl"), os.path.abspath("envs/pushing_env/single_subgoal/goal-subgoal-problem.pddl"))
