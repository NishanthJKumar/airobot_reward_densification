import numpy as np
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

    def neighbors(self, env, loc1, loc2):
        if "loc" in loc1 and "loc" in loc2:
            loc1_index = int(loc1[len("loc"):])
            loc2_index = int(loc2[len("loc"):])
            if env._granularity % 2 == 0:
                rows, cols = env._granularity, env._granularity
            else:
                rows, cols = env._granularity - 1, 2 ** env._granularity / (env._granularity - 1)
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
    
    def get_predicates(self):
        return {"0-arity": [], "1-arity": [self.is_goal], "2-arity": [self.at, self.neighbors]}


def get_state_grounded_atoms(env):
    state_grounded_atoms = []

    predicates = Predicates().get_predicates()
    loc_objects = [f"loc{i}" for i in env._granularity ** 2]
    objects = ['claw', 'goal'] + loc_objects
    for predicate in predicates["0-arity"]:
        state_grounded_atoms.append([(predicate.__name__,), predicate(env)])

    for predicate in predicates["1-arity"]:
        for obj in objects:
            if obj in ['goal'] + loc_objects:
                state_grounded_atoms.append([(predicate.__name__, obj), predicate(env, obj)])
    
    for predicate in predicates["2-arity"]:
        for obj1 in ['claw']:
            for obj2 in  ['goal'] + loc_objects:
                state_grounded_atoms.append([(predicate.__name__, obj1, obj2), predicate(env, obj1, obj2)]) 

    return [atom[0] for atom in state_grounded_atoms if atom[1]]