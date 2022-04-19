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

def play_video(video_dir, video_file=None, play_rate=0.2):
    if video_file is None:
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob(f"**/render_video.mp4"))
        video_files.sort()
        video_file = video_files[-1]
    else:
        video_file = Path(video_file)
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
    ):
        self._action_repeat = action_repeat
        self._max_episode_length = max_episode_length
        self._dist_threshold = dist_threshold
        self._with_obstacle = with_obstacle
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
        self._subgoal2_pos = np.array([[0.24, 0.15, 1.0], [0.76, 0.15, 1.0]])
        self._subgoal1_pos = np.array([[0.36, -0.3, 1.0], [0.64, -0.3, 1.0]])
        self._subgoal_urdf_id = []
        for pos in self._subgoal_pos:
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
        dist_to_goal = np.linalg.norm(state - self._goal_pos[:2])
        success = dist_to_goal < self._dist_threshold
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


module_name = __name__

env_name = "URReacher-v1"
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f"{module_name}:URRobotGym",
)

# DO NOT MODIFY THIS
def train_ppo(
    with_obstacle=False,
    push_exp=False,
    max_steps=200000,
):
    set_config("ppo")
    cfg.alg.num_envs = 1
    cfg.alg.episode_steps = 100
    cfg.alg.max_steps = max_steps
    cfg.alg.deque_size = 20
    cfg.alg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.alg.env_name = "URPusher-v1" if push_exp else "URReacher-v1"
    cfg.alg.save_dir = Path.cwd().absolute().joinpath("data").as_posix()
    cfg.alg.save_dir += "/"
    if push_exp:
        cfg.alg.save_dir += "push"
    else:
        cfg.alg.save_dir += f"ob_{str(with_obstacle)}"
        cfg.alg.save_dir += str(cfg.alg.seed)
    setattr(cfg.alg, "diff_cfg", dict(save_dir=cfg.alg.save_dir))

    print(f"====================================")
    print(f"      Device:{cfg.alg.device}")
    print(f"      Total number of steps:{cfg.alg.max_steps}")
    print(f"====================================")

    set_random_seed(cfg.alg.seed)
    env_kwargs = (
        dict(
            with_obstacle=with_obstacle,
        )
        if not push_exp
        else dict()
    )
    env = make_vec_env(
        cfg.alg.env_name, cfg.alg.num_envs, seed=cfg.alg.seed, env_kwargs=env_kwargs
    )
    env.reset()
    ob_size = env.observation_space.shape[0]

    actor_body = MLP(
        input_size=ob_size,
        hidden_sizes=[64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh,
    )

    critic_body = MLP(
        input_size=ob_size,
        hidden_sizes=[64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh,
    )
    if isinstance(env.action_space, gym.spaces.Discrete):
        act_size = env.action_space.n
        actor = CategoricalPolicy(actor_body, in_features=64, action_dim=act_size)
    elif isinstance(env.action_space, gym.spaces.Box):
        act_size = env.action_space.shape[0]
        actor = DiagGaussianPolicy(
            actor_body,
            in_features=64,
            action_dim=act_size,
            tanh_on_dist=cfg.alg.tanh_on_dist,
            std_cond_in=cfg.alg.std_cond_in,
        )
    else:
        raise TypeError(f"Unknown action space type: {env.action_space}")

    critic = ValueNet(critic_body, in_features=64)
    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = ShapedRewardEpisodicRunner("sas_plan.1", agent=agent, env=env)
    engine = PPOEngine(agent=agent, runner=runner)
    engine.train()
    stat_info, _ = engine.eval(
        render=False, save_eval_traj=True, eval_num=1, sleep_time=0.0
    )
    pprint.pprint(stat_info)
    return cfg.alg.save_dir


# call train_ppo, just set the argument flag properly
save_dir = train_ppo(
    with_obstacle=True,
    push_exp=False,
    max_steps=200000,
)
play_video(save_dir)
#### TODO: plot return and success rate curves
# steps, returns, success_rate = read_tf_log(save_dir)
# data_dict = {}
# data_dict["return"] = [steps, returns]
# plot_curves(data_dict, "Reaching Task without Obstacles - Sparse Reward")
# data_dict = {}
# data_dict["success_rate"] = [steps, success_rate]
# plot_curves(data_dict, "Reaching Task without Obstacles - Sparse Reward")
