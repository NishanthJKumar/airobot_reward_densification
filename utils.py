from typing import List
import pddlpy
import numpy as np
import copy
import os
import glob

max_plan_step_reached = 0

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

    def __init__(self, domain_file_path, problem_file_path, vec_env, classifiers, path_to_fd_folder, task_success_fn):
        self.domain_file_path = domain_file_path
        # TODO: make problem file automatically instead of taking in
        # right now. This can be done by just running the classifiers
        # on the vec_env and generating the initial state!
        self.problem_file_path = problem_file_path
        self.vec_env = vec_env
        self.classifiers = classifiers
        self.task_success_fn = task_success_fn

        # NOTE: In the future, we should be generaitng this problem_file_path
        # within this init method.
        self.domprob = pddlpy.DomainProblem(domain_file_path, problem_file_path)
        os.system(f'python {path_to_fd_folder}/fast-downward.py --alias seq-sat-lama-2011 {domain_file_path} {problem_file_path}')
        plan_file_name = "sas_plan.1"
        with open(plan_file_name) as f:
            # TODO (wmcclinton) automatically genetate plan_file from folder
            self.plan = [eval(line.replace('\n','').replace(' ','\', \'').replace('(','(\'').replace(')','\')')) for line in f.readlines() if 'unit cost' not in line]
        
    def reset_max_plan_step_reached(self):
        global max_plan_step_reached
        max_plan_step_reached = 0

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
                    if obj1 == obj2:
                        continue
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
        global max_plan_step_reached
        for i, grounded_atoms in enumerate(plan[max_plan_step_reached:]):
            # NOTE: using set() is very important here to remove potential duplicates
            # and make the comparison agnostic to order!
            if set(grounded_atoms) == set(state_grounded_atoms):
                return i + max_plan_step_reached
        return max_plan_step_reached

    def get_shaped_reward(self, env, state, previous_state_grounded_atoms, next_state_grounded_atoms, plan):
        global max_plan_step_reached
        success = self.task_success_fn(env, state)
        reward = 1 if success else 0

        prev_phi = self.phi(previous_state_grounded_atoms, plan)
        if max_plan_step_reached < prev_phi:
            max_plan_step_reached = prev_phi
            # import ipdb; ipdb.set_trace()

            # if max_plan_step_reached >= 9:
            #     print(env._t)
            #     print(dist_to_goal)
            #     import ipdb; ipdb.set_trace()

        f = self.phi(next_state_grounded_atoms, plan) - self.phi(previous_state_grounded_atoms, plan)
        reward = reward + f
        # reward = -dist_to_goal
        info = dict(success=success)

        # if dist_to_goal <= 0.1:
        #     print(env._t)
        #     import ipdb; ipdb.set_trace()

        return reward, info

