import os
import numpy as np
from cachetools import cached
from cachetools.keys import hashkey
from envs.base_classifiers_class import BaseClassifiers

class PushingGridBasedClassifiers(BaseClassifiers):
    def robot_at(self, env, gripper, loc):
        if (gripper != "claw"):
            return False
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        elif "loc" in loc:
            loc_index = int(loc[len("loc"):])
            if (loc_index > 2 ** env._granularity - 1):
                raise ValueError(f"loc ranges from 0 to {2 ** env._granularity - 1} but {loc} was passed!")
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
            return (x_lower_bound <= env.robot.arm.get_ee_pose()[0][0] <= x_upper_bound) and (y_lower_bound <= env.robot.arm.get_ee_pose()[0][1] <= y_upper_bound)
        else:
            raise ValueError(f"loc should be either 'goal' or must start with 'loc' not '{loc}'")
    
    def object_at(self, env, obj, loc):
        object_pos = np.zeros(2)
        if (obj == "box1"):
            object_pos, _ = env.robot.pb_client.get_body_state(env._box_id)[:2] 
        else:
            return False
        if self.is_goal(env, loc):
            return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - env._goal_pos[:2]) < env._dist_threshold
        elif "loc" in loc:
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
            return (x_lower_bound <= object_pos[0] <= x_upper_bound) and (y_lower_bound <= object_pos[1] <= y_upper_bound)
        else:
            raise ValueError(f"loc should be either 'goal' or must start with 'loc' not '{loc}'")

    def is_goal(self, env, loc):
        return loc == "goal"

    def occupied(self, env, entity, loc):
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

        if entity == "goal":
            goal_pos_x, goal_pos_y = env._goal_pos[:2]
            if (x_lower_bound <= goal_pos_x <= x_upper_bound and y_lower_bound <= goal_pos_y <= y_upper_bound):
                return True
        else:
            raise Exception(f"Entity should be 'goal', not {entity}")
        return False

    # NOTE: Important to make things efficient! Every time we have a constant
    # predicate, we should cache it.
    @cached(cache={}, key = lambda self, env, loc1, loc2: hashkey(loc1, loc2))
    def neighbors(self, env, loc1, loc2):
        if "loc" in loc1 and "loc" in loc2:
            loc1_index = int(loc1[len("loc"):])
            loc2_index = int(loc2[len("loc"):])
            if env._granularity % 2 == 0:
                square = int(np.sqrt(2 ** env._granularity))
                rows, cols = square, square
            else:
                square = int(np.sqrt(2 ** (env._granularity - 1)))
                rows, cols = square, int((2 ** env._granularity) / square)
            loc1_x, loc1_y = loc1_index // cols, loc1_index % cols
            loc2_x, loc2_y = loc2_index // cols, loc2_index % cols

            # Check that the two locations are in the same column and
            # adjacent rows.
            if loc1_x == loc2_x:
                return loc1_y == loc2_y - 1 or loc1_y == loc2_y + 1

            # Check that the two locations are in the same row and
            # adjacent columns.
            if loc1_y == loc2_y:
                return loc1_x == loc2_x - 1 or loc1_x == loc2_x + 1

            return False
        else:
            if loc1 == "goal" and self.occupied(env, loc1, loc2):
                return True
            elif loc2 == "goal" and self.occupied(env, loc2, loc1):
                return True
            else:
                return False

    def get_typed_predicates(self):
        return {"0-arity": [], "1-arity": [(self.is_goal, "location")], "2-arity": [(self.robot_at, "gripper", "location"), (self.object_at, "box", "location"), (self.neighbors, "location", "location")]}

    def get_path_to_domain_and_problem_files(self):
        return (os.path.abspath("envs/pushing_env/grid_based/pushing-grid-domain.pddl"), os.path.abspath("envs/pushing_env/grid_based/pushing-grid-problem6.pddl"))
