import pddlpy
import numpy as np

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
            wall_min_x, wall_min_y = 0.5 - (0.18/2), 0.15 - (0.01/2)
            wall_max_x, wall_max_y = 0.5 + (0.18/2), 0.15 + (0.01/2)
            if (x_upper_bound < wall_min_x or x_lower_bound > wall_max_x) or (y_upper_bound < wall_min_y or y_lower_bound > wall_max_y):     
                return False
            elif (wall_min_x < x_lower_bound < wall_max_x or wall_min_x < x_upper_bound < wall_max_x):
                if (y_upper_bound < wall_min_y or y_lower_bound > wall_max_y):
                    return False
                else:
                    return True
            else:
                return True
        elif entity == "goal":
            goal_pos_x, goal_pos_y = env._goal_pos[:2]
            if (x_lower_bound <= goal_pos_x <= x_upper_bound and y_lower_bound <= goal_pos_y <= y_upper_bound):
                return True
        else:
            raise Exception(f"Entity should be either 'obstacle' or 'goal', not {entity}")

        return False

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

    def get_predicates(self):
        return {"0-arity": [], "1-arity": [self.is_goal], "2-arity": [self.at, self.neighbors]}


def get_state_grounded_atoms(env):
    state_grounded_atoms = []

    predicates = Predicates().get_predicates()
    loc_objects = [f"loc{i}" for i in range(2 ** env._granularity)]
    objects = ['claw', 'goal'] + loc_objects
    for predicate in predicates["0-arity"]:
        state_grounded_atoms.append([(predicate.__name__,), predicate(env)])

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj in ['goal'] + loc_objects:
                state_grounded_atoms.append([(predicate.__name__, obj), predicate(env, obj)])

    # print("loc_objects = ", loc_objects)
    #TODO (vp): This needs to change since the neighbors predicate is over locations and not claw.
    for predicate in predicates["2-arity"]:
        if predicate.__name__ == "at":
            for obj in ['goal'] + loc_objects:
                state_grounded_atoms.append([(predicate.__name__, "claw", obj), predicate(env, "claw", obj)])
        elif predicate.__name__ == "neighbors":
            for obj1 in loc_objects:
                for obj2 in loc_objects:
                    state_grounded_atoms.append([(predicate.__name__, obj1, obj2), predicate(env, obj1, obj2)])
            for obj in loc_objects:
                state_grounded_atoms.append([(predicate.__name__, obj, "goal"), predicate(env, obj, "goal")])
                state_grounded_atoms.append([(predicate.__name__, "goal", obj), predicate(env, "goal", obj)])

    return [atom[0] for atom in state_grounded_atoms if atom[1]]
