import time
from copy import deepcopy

import numpy as np
import torch
from easyrl.runner.base_runner import BasicRunner
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.gym_util import get_render_images
from easyrl.utils.torch_util import torch_to_np
from utils import Predicates, apply_grounded_operator, get_state_grounded_atoms, apply_grounded_plan, get_shaped_reward

class ShapedRewardEpisodicRunner(BasicRunner):
    """
    This only applies to environments that are wrapped by VecEnv.
    It assumes the environment is automatically reset if done=True
    """

    def __init__(self, plan_file_name, *args, **kwargs):
        super(ShapedRewardEpisodicRunner, self).__init__(*args, **kwargs)
        with open(plan_file_name) as f:
            self.plan = [eval(line.replace('\n','').replace(' ','\', \'').replace('(','(\'').replace(')','\')')) for line in f.readlines() if 'unit cost' not in line]

    @torch.no_grad()
    def __call__(self, time_steps, sample=True, evaluation=False,
                 return_on_done=False, render=False, render_image=False,
                 sleep_time=0, reset_first=False,
                 reset_kwargs=None, action_kwargs=None,
                 random_action=False, get_last_val=False):
        traj = Trajectory()
        if reset_kwargs is None:
            reset_kwargs = {}
        if action_kwargs is None:
            action_kwargs = {}
        if evaluation:
            env = self.eval_env
        else:
            env = self.train_env
        if self.obs is None or reset_first or evaluation:
            self.reset(env=env, **reset_kwargs)
        ob = self.obs  #['observation'] # NOTE: (njk) this is necessary for FetchBlockConstructionEnv
        # this is critical for some environments depending
        # on the returned ob data. use deepcopy() to avoid
        # adding the same ob to the traj

        # only add deepcopy() when a new ob is generated
        # so that traj[t].next_ob is still the same instance as traj[t+1].ob
        ob = deepcopy(ob)
        if return_on_done:
            all_dones = np.zeros(env.num_envs, dtype=bool)
        else:
            all_dones = None
        for t in range(time_steps):
            if render:
                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if render_image:
                # get render images at the same time step as ob
                imgs = get_render_images(env)

            if random_action:
                action = env.random_actions()
                action_info = dict()
            else:
                action, action_info = self.agent.get_action(ob,
                                                            sample=sample,
                                                            **action_kwargs)

            previous_state_grounded_atoms = get_state_grounded_atoms(env.envs[0])
            next_ob, _, done, env_info = env.step(action)
            next_state_grounded_atoms = get_state_grounded_atoms(env.envs[0])
            plan_grounded_atoms = apply_grounded_plan(previous_state_grounded_atoms, self.plan)
            reward, info = get_shaped_reward(env.envs[0], next_ob, previous_state_grounded_atoms, next_state_grounded_atoms, plan_grounded_atoms)
            reward = np.array([reward])
            info.update(env_info[0])
            info = [info]
            
            next_ob = next_ob#['observation']

            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            true_next_ob, true_done, all_dones = self.get_true_done_next_ob(next_ob,
                                                                            done,
                                                                            reward,
                                                                            info,
                                                                            all_dones,
                                                                            skip_record=evaluation)
            sd = StepData(ob=ob,
                          action=action,
                          action_info=action_info,
                          next_ob=true_next_ob,
                          reward=reward,
                          done=true_done,
                          info=info)
            ob = next_ob
            traj.add(sd)
            if return_on_done and np.all(all_dones):
                break

        if get_last_val and not evaluation:
            last_val = self.agent.get_val(traj[-1].next_ob)
            traj.add_extra('last_val', torch_to_np(last_val))
        self.obs = ob if not evaluation else None
        return traj
