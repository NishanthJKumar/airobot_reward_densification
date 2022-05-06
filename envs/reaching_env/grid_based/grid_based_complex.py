from envs.base_classifiers_class import BaseClassifiers
from cachetools import cached
from cachetools.keys import hashkey
import numpy as np
import os

class ReachingGridBasedComplexClassifiers(BaseClassifiers):
    def at(self, env, gripper, loc):
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
            # x_mid, y_mid = (x_lower_bound + x_upper_bound) / 2, (y_lower_bound + y_upper_bound) / 2
            # return np.linalg.norm(env.robot.arm.get_ee_pose()[0][:2] - np.array([x_mid, y_mid])) < env._dist_threshold
            at_location =  (x_lower_bound <= env.robot.arm.get_ee_pose()[0][0] <= x_upper_bound) and (y_lower_bound <= env.robot.arm.get_ee_pose()[0][1] <= y_upper_bound)
            return at_location
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

        if entity == "obstacle":
            
            # Check collision with wall 1 - vertical wall
            wall_min_x_1, wall_min_y_1 = 0.5 - 0.1, -0.2 - 0.01
            wall_max_x_1, wall_max_y_1 = 0.5 + 0.1, -0.2 + 0.01
            if (wall_min_x_1 < x_lower_bound < wall_max_x_1 or wall_min_x_1 < x_upper_bound < wall_max_x_1):
                if not (y_upper_bound < wall_min_y_1 or y_lower_bound > wall_max_y_1):
                    return True
                
            # Check collision with wall 2 - top horizontal wall 
            wall_min_x_2, wall_min_y_2 = 0.4 - 0.01, -0.03 - 0.18
            wall_max_x_2, wall_max_y_2 = 0.4 + 0.01, -0.03 + 0.18
            if (wall_min_y_2 < y_lower_bound < wall_max_y_2 or wall_min_y_2 < y_upper_bound < wall_max_y_2):
                if not (x_upper_bound < wall_min_x_2 or x_lower_bound > wall_max_x_2):
                    return True
            
            # Check collision with wall 3 - bottom horizontal wall
            wall_min_x_3, wall_min_y_3 = 0.6 - 0.01, -0.03 - 0.18
            wall_max_x_3, wall_max_y_3 = 0.6 + 0.01, -0.03 + 0.18
            if (wall_min_y_3 < y_lower_bound < wall_max_y_3 or wall_min_y_3 < y_upper_bound < wall_max_y_3):
                if not (x_upper_bound < wall_min_x_3 or x_lower_bound > wall_max_x_3):
                    return True
            
            return False
        elif entity == "goal":
            goal_pos_x, goal_pos_y = env._goal_pos[:2]
            if (x_lower_bound <= goal_pos_x <= x_upper_bound and y_lower_bound <= goal_pos_y <= y_upper_bound):
                return True
        else:
            raise Exception(f"Entity should be either 'obstacle' or 'goal', not {entity}")

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

            if self.occupied(env, "obstacle", loc1) or self.occupied(env, "obstacle", loc2):
                return False

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
        return {"0-arity": [], "1-arity": [(self.is_goal, "location")], "2-arity": [(self.at, "gripper", "location"), (self.neighbors, "location", "location")]}

    def get_path_to_domain_and_problem_files(self):
        return (os.path.abspath("envs/reaching_env/grid_based/reaching-grid-domain.pddl"), os.path.abspath("envs/reaching_env/grid_based/reaching-grid-complex-problem6.pddl"))
