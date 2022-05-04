import time
from copy import deepcopy

import numpy as np
import torch
from easyrl.runner.base_runner import BasicRunner
from easyrl.configs import cfg
from easyrl.configs.sac_config import SACConfig
from easyrl.utils.data import StepData
from easyrl.utils.data import Trajectory
from easyrl.utils.gym_util import get_render_images
from easyrl.utils.torch_util import torch_to_np
import cv2
import os

class ShapedRewardEpisodicRunner(BasicRunner):
    """
    This only applies to environments that are wrapped by VecEnv.
    It assumes the environment is automatically reset if done=True
    """

    def __init__(self, g_utils, dynamic_reward_shaping, *args, **kwargs):
        super(ShapedRewardEpisodicRunner, self).__init__(*args, **kwargs)
        self.g_utils = g_utils
        self.dynamic_reward_shaping = dynamic_reward_shaping
        self.plan = self.g_utils.plan
        self.plan_grounded_atoms = None
        if cfg.alg.epsilon is not None:
            np.random.seed(cfg.alg.seed)
            self.epsilon = cfg.alg.epsilon
            self.epsilon_reduction = self.epsilon / cfg.alg.max_steps
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
        ob = self.obs
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

            if cfg.alg.epsilon is not None and not evaluation:
                if np.random.uniform() < self.epsilon:
                    if isinstance(cfg, SACConfig):
                        random_action = True
                    else:
                        sample = True
                else:
                    if isinstance(cfg, SACConfig):
                        random_action = False
                    else:
                        sample = False

            if random_action:
                action = env.random_actions()
                action_info = dict()
            else:
                action, action_info = self.agent.get_action(ob,
                                                            sample=sample,
                                                            **action_kwargs)

            previous_state_grounded_atoms = self.g_utils.get_state_grounded_atoms(env.envs[0])
            if self.plan_grounded_atoms is None:
                # This is the first time we're calling the function, so
                # we can compute the plan_grounded_atoms.
                self.plan_grounded_atoms = self.g_utils.apply_grounded_plan(previous_state_grounded_atoms, self.plan)
            next_ob, reward, done, env_info = env.step(action)
            next_state_grounded_atoms = self.g_utils.get_state_grounded_atoms(env.envs[0])

            if env.envs[0].reward_type == "pddl":
                if env.envs[0]._t != 0:
                    reward, info = self.g_utils.get_shaped_reward(env.envs[0], ob, next_ob, previous_state_grounded_atoms, next_state_grounded_atoms, self.plan_grounded_atoms, self.dynamic_reward_shaping)
                    reward = np.array([reward])
                else:
                    reward = np.array([0.0])
                    info = dict(success=False)
                if env.envs[0]._t == 0 and env.envs[0].max_plan_step_reached != 0:
                    import ipdb; ipdb.set_trace()
            else:
                info = dict()

            info.update(env_info[0])
            info = [info]

            # Rendering!
            # if evaluation:
            #     cv2.imshow("img", env.render())
            #     cv2.waitKey(25)
            
            if render_image:
                for img, inf in zip(imgs, info):
                    inf['render_image'] = deepcopy(img)

            # NOTE: true_done tells us whether the goal was actually reached.
            # If done is True but true_done is False, then the episode terminated due
            # to timeout.
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
            if cfg.alg.epsilon is not None and not evaluation:
                self.epsilon = max([0.2, self.epsilon - self.epsilon_reduction])
            if return_on_done and np.all(all_dones):
                break

        if get_last_val and not evaluation:
            last_val = self.agent.get_val(traj[-1].next_ob)
            traj.add_extra('last_val', torch_to_np(last_val))
        self.obs = ob if not evaluation else None

        return traj
