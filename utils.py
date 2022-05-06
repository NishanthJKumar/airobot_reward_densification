from typing import List
import pddlpy
import numpy as np
import copy
import os
import glob
ALPHA = 1
SCALE = 100

def play_video(video_dir, video_file=None, play_rate=0.2):
    if video_file is None:
        video_files = list(glob.glob(video_dir + "/eval/**/render_video.mp4"))
        video_files.sort()
        video_file = video_files[-1]
    else:
        video_file = os.Path(video_file)
    os.system(f"vlc --rate {str(play_rate + 0.01)} {video_file}")

# read tf log file
def read_tf_log(log_dir):
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob(f"**/events.*"))
    if len(log_files) < 1:
        return None
    log_file = log_files[0]
    event_acc = EventAccumulator(log_file.as_posix())
    event_acc.Reload()
    tags = event_acc.Tags()
    try:
        scalar_success = event_acc.Scalars("train/episode_success")
        success_rate = [x.value for x in scalar_success]
        steps = [x.step for x in scalar_success]
        scalar_return = event_acc.Scalars("train/episode_return/mean")
        returns = [x.value for x in scalar_return]
    except:
        return None
    return steps, returns, success_rate

def plot_curves(data_dict, title):
    # {label: [x, y]}
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = data_dict.keys()
    for label, data in data_dict.items():
        x = data[0]
        y = data[1]
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.legend()

class GroundingUtils:
    def __init__(self, domain_file_path, problem_file_path, vec_env, classifiers, path_to_fd_folder, task_success_fn, pddl_type, env_type):
        self.domain_file_path = domain_file_path
        # TODO: make problem file automatically instead of taking in
        # right now. This can be done by just running the classifiers
        # on the vec_env and generating the initial state!
        self.problem_file_path = problem_file_path
        self.vec_env = vec_env
        self.classifiers = classifiers
        self.task_success_fn = task_success_fn
        self.pddl_type = pddl_type
        self.env_type = env_type

        # NOTE: In the future, we should be generaitng this problem_file_path
        # within this init method.
        self.domprob = pddlpy.DomainProblem(domain_file_path, problem_file_path)
        os.system(f'python {path_to_fd_folder}/fast-downward.py --alias seq-sat-lama-2011 {domain_file_path} {problem_file_path} >/dev/null 2>&1')
        plan_file_name = "sas_plan.1"
        with open(plan_file_name) as f:
            # TODO (wmcclinton) automatically genetate plan_file from folder
            self.plan = [eval(line.replace('\n','').replace(' ','\', \'').replace('(','(\'').replace(')','\')')) for line in f.readlines() if 'unit cost' not in line]

    def get_state_grounded_atoms(self, env):
        state_grounded_atoms = []
        predicates = self.classifiers.get_typed_predicates()
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
                    if obj1 == obj2:
                        continue
                    if obj1[1] == predicate[1] and obj2[1] == predicate[2]:
                        state_grounded_atoms.append([(predicate[0].__name__, obj1[0], obj2[0]), predicate[0](env, obj1[0], obj2[0])]) 
        return [atom[0] for atom in state_grounded_atoms if atom[1]]
    
    def apply_grounded_operator(self, state_grounded_atoms, op_name, params):
        for o in self.domprob.ground_operator(op_name):
            if set(params) == set(o.variable_list.values()) and o.precondition_pos.issubset(state_grounded_atoms):
                next_state_grounded_atoms = copy.deepcopy(state_grounded_atoms)
                for effect in o.effect_pos:
                    next_state_grounded_atoms.append(effect)
                for effect in o.effect_neg:
                    next_state_grounded_atoms.remove(effect)
                return next_state_grounded_atoms
        import ipdb; ipdb.set_trace()
        raise ValueError("Couldn't compute next_state_grounded_atoms")
    
    def apply_grounded_plan(self, state_grounded_atoms, plan):
        plan_grounded_atoms = []
        plan_grounded_atoms.append(state_grounded_atoms)
        for ground_operator in plan:
            op_name = ground_operator[0]
            params = list([ground_operator[1]] if len(ground_operator[1]) == 2 else ground_operator[1:])
            # import ipdb; ipdb.set_trace()
            plan_grounded_atoms.append(self.apply_grounded_operator(plan_grounded_atoms[-1], op_name, params))
        return plan_grounded_atoms
    
    def phi_t(self, t):
        return (1/t)
    
    def inv_phi_t(self, t):
        return t
    
    def loc2xy(self, env, loc_index):
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
        return np.array([(x_upper_bound + x_lower_bound)/2, (y_upper_bound + y_lower_bound)/2])
    
    def get_dist_to_next_subgoal(self, env, state, next_subgoal):
        distance = 0
        for predicate in next_subgoal:
            if (predicate[0] == "at" and predicate[1] == "claw" and self.env_type == "reach") or \
            ("at" in predicate[0] and predicate[1] == "box1" and self.env_type == "push"):
                if self.env_type == "reach":
                    obj_state = state[0][:2]
                elif self.env_type == "push":
                    state_grounded_atoms = self.get_state_grounded_atoms(env)
                    predicate_diff = list(set(next_subgoal) - set(state_grounded_atoms))
                    if len(predicate_diff) > 0:
                        predicate = predicate_diff[0]
                        if predicate[1] == 'claw':
                            obj_state = state[0][:2]
                        elif predicate[1] == 'box1':
                            obj_state = state[0][2:]
                        else:
                            raise NotImplementedError("FML")
                    else:
                        obj_state = state[0][2:]
                if self.pddl_type == "single_subgoal":
                    # SINGLE SUBGOAL
                    if self.env_type == "reach" and predicate[2] == 'subgoal':
                        nextgoal_xy = env._subgoal2_pos[0][:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return distance
                    elif self.env_type == "push" and predicate[2] == 'subgoal2':
                        nextgoal_xy = env._subgoal2_pos[:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return distance
                    else:
                        nextgoal_xy = env._goal_pos[:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return 2 * distance
                    
                elif self.pddl_type == "multi_subgoal":
                    # MULTI SUBGOAL
                    if predicate[2] == 'subgoal1' and self.env_type == "reach":
                        nextgoal_xy = env._subgoal1_pos[0][:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return distance
                    elif predicate[2] == 'subgoal1' and self.env_type == "push":
                        nextgoal_xy = env._subgoal1_pos[:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return distance
                    elif predicate[2] == 'subgoal2' and self.env_type == "reach":
                        nextgoal_xy = env._subgoal2_pos[0][:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return 2 * distance
                    elif predicate[2] == 'subgoal2' and self.env_type == "push":
                        nextgoal_xy = env._subgoal2_pos[:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return 2 * distance
                    elif predicate[2] == 'subgoal3' and self.env_type == "reach":
                        nextgoal_xy = env._subgoal3_pos[0][:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return 10 * distance
                    elif predicate[2] == 'subgoal3' and self.env_type == "push":
                        nextgoal_xy = env._subgoal3_pos[:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return 10 * distance
                    else:
                        nextgoal_xy = env._goal_pos[:2]
                        distance = np.linalg.norm(nextgoal_xy - obj_state)
                        return 10 * distance
                
                elif self.pddl_type == "grid_based":
                    # GRID-BASED
                    nextgoal_xy = self.loc2xy(env, int(predicate[2].replace("loc","")))
                    #print(nextgoal_xy)
                    distance = (env.max_plan_step_reached + 1) * np.linalg.norm(nextgoal_xy - obj_state)
                    return distance

        raise NotImplementedError("get_dist_to_next_subgoal not implemented for this environment")
    
    def dist_phi(self, env, state, next_subgoal):
        return -1 * self.get_dist_to_next_subgoal(env, state, next_subgoal)
    
    def phi(self, env, state_grounded_atoms, plan, dynamic_reward_shaping, state=None):
        t = env._t
        for i, grounded_atoms in enumerate(plan[env.max_plan_step_reached:]):
            # NOTE: using set() is very important here to remove potential duplicates
            # and make the comparison agnostic to order!
            if set(grounded_atoms) == set(state_grounded_atoms):
                if dynamic_reward_shaping is None:
                    return i + env.max_plan_step_reached
                elif dynamic_reward_shaping == "basic":
                    if t == 0:
                        return [i + env.max_plan_step_reached, i + env.max_plan_step_reached]
                    # Returns phi(s, t) and phi(s) which is used to update max_plan_step_reached
                    return [(i + env.max_plan_step_reached) * self.phi_t(t), i + env.max_plan_step_reached]
                elif dynamic_reward_shaping == "dist":
                    # Returns phi(s, t) and phi(s) which is used to update max_plan_step_reached
                    if env.max_plan_step_reached + 1 > len(plan) - 1:
                        return [ALPHA * self.dist_phi(env, state, plan[-1]), i + env.max_plan_step_reached] #* self.phi_t(t)
                    else:
                        return [ALPHA * self.dist_phi(env, state, plan[env.max_plan_step_reached + 1]), i + env.max_plan_step_reached] #* self.phi_t(t)
                else:
                    raise NotImplementedError(f"{dynamic_reward_shaping} is not a valid dynamic reward shaping function")
        if dynamic_reward_shaping is None:
            return env.max_plan_step_reached
        elif dynamic_reward_shaping == "basic":
            # In basic reward shaping you cannot divide by 1/t when t=0 so just returns 0
            if t == 0:
                return [env.max_plan_step_reached, env.max_plan_step_reached]
            # Returns phi(s, t) and phi(s) which is used to update max_plan_step_reached
            return [(env.max_plan_step_reached) * self.phi_t(t), env.max_plan_step_reached]
        elif dynamic_reward_shaping == "dist":
            # Returns phi(s, t) and phi(s) which is used to update max_plan_step_reached
            if env.max_plan_step_reached + 1 > len(plan) - 1:
                return [ALPHA * self.dist_phi(env, state, plan[-1]), i + env.max_plan_step_reached] #* self.phi_t(t)
            else:
                return [ALPHA * self.dist_phi(env, state, plan[env.max_plan_step_reached + 1]), env.max_plan_step_reached] #* self.phi_t(t)
        else:
            raise NotImplementedError(f"{dynamic_reward_shaping} is not a valid dynamic reward shaping function")
    
    def get_shaped_reward(self, env, prev_state, state, previous_state_grounded_atoms, next_state_grounded_atoms, plan, dynamic_reward_shaping):
        success = self.task_success_fn(env, state)
        reward = 1 if success else 0
        if dynamic_reward_shaping is not None:
            # Set prev_phi to phi(s) to update max_plan_step_reached
            prev_phi = self.phi(env, previous_state_grounded_atoms, plan, dynamic_reward_shaping, state=state)[1]
        else:
            prev_phi = self.phi(env, previous_state_grounded_atoms, plan, dynamic_reward_shaping)
    
        if env.max_plan_step_reached < prev_phi:
            env.max_plan_step_reached = prev_phi
            
            # if max_plan_step_reached >= 9:
            #     print(env._t)
            #     print(dist_to_goal)
            #     import ipdb; ipdb.set_trace()
        if dynamic_reward_shaping is not None:
            # Computes F using phi(s,t)
            f = self.phi(env, next_state_grounded_atoms, plan, dynamic_reward_shaping, state=state)[0] - self.phi(env, previous_state_grounded_atoms, plan, dynamic_reward_shaping, state=prev_state)[0]
        else:
            # Computes F using phi(s)
            f = self.phi(env, next_state_grounded_atoms, plan, dynamic_reward_shaping) - self.phi(env, previous_state_grounded_atoms, plan, dynamic_reward_shaping)
        reward = SCALE * (reward + f)
        info = dict(success=success)
        return reward, info
