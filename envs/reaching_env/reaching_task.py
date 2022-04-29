import os
import torch
import gym
import pprint
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from airobot import Robot
from airobot.utils.common import euler2quat
from gym import spaces
from gym.envs.registration import registry, register
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from pathlib import Path
from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg, set_config
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.utils.common import set_random_seed, load_from_json
from easyrl.utils.gym_util import make_vec_env
from base64 import b64encode
from shaped_reward_episodic_runner import ShapedRewardEpisodicRunner
from IPython import display
from IPython.display import HTML


def check_collision_rate(log_dir):
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob(f"**/info.json"))
    log_files.sort()
    log_file = log_files[-1]
    info_data = load_from_json(log_file)
    collisions = [v["collision"] for k, v in info_data.items()]
    return np.mean(collisions)

class URRobotGym(gym.Env):
    def __init__(
        self,
        action_repeat=10,
        with_obstacle=False,
        # Set 'gui' to False if you are using Colab, otherwise the session will crash as Colab does not support X window
        # You can set it to True for debugging purpose if you are running the notebook on a local machine.
        gui=False,
        max_episode_length=25,
        dist_threshold=0.05,
        granularity=5,
    ):
        self._action_repeat = action_repeat
        self._max_episode_length = max_episode_length
        self._dist_threshold = dist_threshold
        self._with_obstacle = with_obstacle
        self._granularity = granularity
        print(f"================================================")
        print(f"With obstacle in the scene:{self._with_obstacle}")
        print(f"================================================")

        self._xy_bounds = np.array(
            [[0.23, 0.78], [-0.35, 0.3]]  # [xmin, xmax]
        )  # [ymin, ymax]
        self.robot = Robot(
            "ur5e_stick",
            pb_cfg={
                "gui": gui,
                "realtime": False,
                "opengl_render": torch.cuda.is_available(),
            },
        )
        self._arm_reset_pos = np.array(
            [
                -0.38337763,
                -2.02650575,
                -2.01989619,
                -0.64477803,
                1.571439041,
                -0.38331266,
            ]
        )
        self._table_id = self.robot.pb_client.load_urdf(
            "table/table.urdf",
            [0.5, 0, 0.4],
            euler2quat([0, 0, np.pi / 2]),
            scaling=0.9,
        )

        # create a ball at the start location (for visualization purpose)
        self._start_pos = np.array([0.45, -0.32, 1.0])
        self._start_urdf_id = self.robot.pb_client.load_geom(
            "sphere", size=0.04, mass=0, base_pos=self._start_pos, rgba=[1, 1, 0, 0.8]
        )

        # create a ball at the goal location
        self._goal_pos = np.array([0.5, 0.26, 1.0])
        self._goal_urdf_id = self.robot.pb_client.load_geom(
            "sphere", size=0.04, mass=0, base_pos=self._goal_pos, rgba=[1, 0, 0, 0.8]
        )

        # disable the collision checking between the robot and the ball at the goal location
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            self.robot.pb_client.setCollisionFilterPair(
                self.robot.arm.robot_id, self._goal_urdf_id, i, -1, enableCollision=0
            )
        # disable the collision checking between the robot and the ball at the start location
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            self.robot.pb_client.setCollisionFilterPair(
                self.robot.arm.robot_id, self._start_urdf_id, i, -1, enableCollision=0
            )

        # create an obstacle
        if self._with_obstacle:
            self._wall_id = self.robot.pb_client.load_geom(
                "box",
                size=[0.18, 0.01, 0.1],
                mass=0,
                base_pos=[0.5, 0.15, 1.0],
                rgba=[0.5, 0.5, 0.5, 0.8],
            )

        # create balls at subgoal locations
        self._subgoal3_pos = np.array([[0.36, 0.18, 1.0], [0.64, 0.18, 1.0]])
        self._subgoal2_pos = np.array([[0.23, 0.15, 1.0], [0.76, 0.15, 1.0]])
        self._subgoal1_pos = np.array([[0.36, 0.0, 1.0], [0.64, 0.0, 1.0]])
        self._subgoal_urdf_id = []
        for pos in self._subgoal1_pos:
            self._subgoal_urdf_id.append(
                self.robot.pb_client.load_geom(
                    "sphere", size=0.04, mass=0, base_pos=pos, rgba=[0, 0.8, 0.8, 0.8]
                )
            )
        for pos in self._subgoal2_pos:
            self._subgoal_urdf_id.append(
                self.robot.pb_client.load_geom(
                    "sphere", size=0.04, mass=0, base_pos=pos, rgba=[0, 0.8, 0.8, 0.8]
                )
            )
        for pos in self._subgoal3_pos:
            self._subgoal_urdf_id.append(
                self.robot.pb_client.load_geom(
                    "sphere", size=0.04, mass=0, base_pos=pos, rgba=[0, 0.8, 0.8, 0.8]
                )
            )
        # disable the collision checking between the robot and the subgoal balls
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            for sg in self._subgoal_urdf_id:
                self.robot.pb_client.setCollisionFilterPair(
                    self.robot.arm.robot_id, sg, i, -1, enableCollision=0
                )

        self._action_bound = 1.0
        self._ee_pos_scale = 0.02
        self._action_high = np.array([self._action_bound] * 2)
        self.action_space = spaces.Box(
            low=-self._action_high, high=self._action_high, dtype=np.float32
        )
        state_low = np.full(len(self._get_obs()), -float("inf"))
        state_high = np.full(len(self._get_obs()), float("inf"))
        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float32)
        self.reset()

    def reset(self):
        self.robot.arm.set_jpos(self._arm_reset_pos, ignore_physics=True)
        self._t = 0
        self._ref_ee_pos = self.robot.arm.get_ee_pose()[0]
        self._ref_ee_ori = self.robot.arm.get_ee_pose()[1]
        return self._get_obs()

    def step(self, action):
        collision = self._apply_action(action)
        self._t += 1
        state = self._get_obs()
        done = self._t >= self._max_episode_length
        reward, info = self._get_reward(
            state=state, action=action, collision=float(collision)
        )
        info["collision"] = collision
        return state, reward, done, info

    def _get_reward(self, state, action, collision):
        reward = None
        success = self.get_success(self, state)
        info = {'success': success}
        return reward, info

    def _get_obs(self):
        gripper_pos = self.robot.arm.get_ee_pose()[0][:2]
        state = gripper_pos
        return state

    def _check_collision_with_wall(self):
        if hasattr(self, "_wall_id"):
            return (
                len(
                    self.robot.pb_client.getContactPoints(
                        self.robot.arm.robot_id, self._wall_id, 10, -1
                    )
                )
                > 0
            )
        else:
            return False

    def _apply_action(self, action):
        jnt_poses = self.robot.arm.get_jpos()
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 2:
            raise ValueError("Action should be [d_x, d_y].")
        # we set dz=0
        action = np.append(action, 0)
        pos, _, _, _ = self.robot.arm.get_ee_pose()
        pos += action[:3] * self._ee_pos_scale
        pos[2] = self._ref_ee_pos[2]
        # if the new position is out of the bounds, then we don't apply the action
        if not np.logical_and(
            np.all(pos[:2] >= self._xy_bounds[:, 0]),
            np.all(pos[:2] <= self._xy_bounds[:, 1]),
        ):
            return False

        # move the end-effector to the new position
        jnt_pos = self.robot.arm.compute_ik(pos, ori=self._ref_ee_ori)
        for step in range(self._action_repeat):
            self.robot.arm.set_jpos(jnt_pos)
            self.robot.pb_client.stepSimulation()

        # if collision occurs, we reset the robot back to its original pose (before apply_action)
        collision = False
        if self._check_collision_with_wall():
            self.robot.arm.set_jpos(jnt_poses, ignore_physics=True)
            collision = True
        return collision

    def render(self, mode, **kwargs):
        robot_base = self.robot.arm.robot_base_pos
        self.robot.cam.setup_camera(
            focus_pt=robot_base, dist=2, yaw=85, pitch=-20, roll=0
        )
        rgb, _ = self.robot.cam.get_images(get_rgb=True, get_depth=False)
        return rgb

    def get_success(self, env, state):
        dist_to_goal = np.linalg.norm(state - env._goal_pos[:2])
        return dist_to_goal < env._dist_threshold


module_name = __name__

env_name = "URReacher-v1"
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f"{module_name}:URRobotGym",
)