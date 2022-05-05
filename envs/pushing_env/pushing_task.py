import gym
import torch
import os
import pddlpy
import numpy as np
from gym import spaces
from airobot import Robot
from pathlib import Path
import matplotlib.pyplot as plt
from airobot.utils.common import euler2quat
from easyrl.utils.common import load_from_json
from gym.envs.registration import registry, register
from envs.pushing_env.grid_based.grid_based import PushingGridBasedClassifiers

def check_collision_rate(log_dir):
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob(f"**/info.json"))
    log_files.sort()
    log_file = log_files[-1]
    info_data = load_from_json(log_file)
    collisions = [v["collision"] for k, v in info_data.items()]
    return np.mean(collisions)

class URRobotPusherGym(gym.Env):
    def __init__(self,
                 action_repeat=10,
                 gui=True,
                 max_episode_length=25,
                 dist_threshold=0.05,
                 granularity=4, 
                 reward_type=None,
                 pddl_type="single_subgoal"): # TODO: Seems like granularity parameter is not set correctly from make_vec_env. Need to check this!
        self._action_repeat = action_repeat
        self.max_plan_step_reached = 0
        self._max_episode_length = max_episode_length
        self._dist_threshold = dist_threshold
        self._granularity = granularity
        self.reward_type = reward_type
        self._pddl_type = pddl_type

        self._xy_bounds = np.array([[0.23, 0.78], # [xmin, xmax]
                                   [-0.35, 0.3]]) # [ymin, ymax]

        self.robot = Robot('ur5e_stick',
                           pb_cfg={'gui': gui,
                                   'realtime': False,
                                   'opengl_render': torch.cuda.is_available()})
        self._arm_reset_pos = np.array([-0.38337763,
                                        -2.02650575,
                                        -2.01989619,
                                        -0.64477803,
                                        1.571439041,
                                        -0.38331266])
        self._table_id = self.robot.pb_client.load_urdf("table/table.urdf",
                                                        [0.5, 0, 0.4],
                                                        euler2quat([0, 0, np.pi / 2]),
                                                        scaling=0.9)

        # create a ball at the start location (for visualization purpose)
        self._start_pos = np.array([0.45, -0.32, 1.0])
        self._start_urdf_id = self.robot.pb_client.load_geom("sphere", size=0.04, mass=0,
                                                             base_pos=self._start_pos,
                                                             rgba=[1, 1, 0, 0.8])

        # create a ball at the goal location
        self._goal_pos = np.array([0.5, 0.2, 1.0])
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

        self._box_pos = np.array([0.35, -0.1, 0.996])
        self._box_id = self.robot.pb_client.load_geom('cylinder', size=[0.05, 0.05], mass=1.,
                                                      base_pos=self._box_pos,
                                                      rgba=[1., 0.6, 0.6, 1])

        self.robot.pb_client.changeDynamics(self._box_id, -1, lateralFriction=0.9)
        self.robot.pb_client.setCollisionFilterPair(self._box_id, self._start_urdf_id, -1, -1, enableCollision=0)
        self.robot.pb_client.setCollisionFilterPair(self._box_id, self._goal_urdf_id, -1, -1, enableCollision=0)

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
        # add the dummy subgoal locations
        self._subgoal_urdf_id = []
        self._subgoal0_pos = self._ref_ee_pos
        self._subgoal1_pos = np.array([0.4, -0.21, 1.0])
        self._subgoal2_pos = self._box_pos
        self._subgoal3_pos = np.array([0.425, 0.05, 1.0])
        if self._pddl_type in ["single_subgoal", "multi_subgoal"]:
            self._subgoal_urdf_id.append(
                self.robot.pb_client.load_geom(
                    "sphere", size=0.04, mass=0, base_pos=self._subgoal2_pos, rgba=[0, 0.8, 0.8, 0.8]
                )
            )
        if self._pddl_type == "multi_subgoal":
            self._subgoal_urdf_id.append(
                self.robot.pb_client.load_geom(
                    "sphere", size=0.04, mass=0, base_pos=self._subgoal1_pos, rgba=[0, 0.8, 0.8, 0.8]
                )
            )
            self._subgoal_urdf_id.append(
                self.robot.pb_client.load_geom(
                    "sphere", size=0.04, mass=0, base_pos=self._subgoal3_pos, rgba=[0, 0.8, 0.8, 0.8]
                )
            )
        
        if self._pddl_type == "grid_based" and not "handcrafted" in self.reward_type:
            domain_file_path, problem_file_path = PushingGridBasedClassifiers().get_path_to_domain_and_problem_files()
            domprob = pddlpy.DomainProblem(domain_file_path, problem_file_path)
            path_to_fd_folder = "/home/njk/Documents/GitHub/downward"
            os.system(f'python {path_to_fd_folder}/fast-downward.py --alias seq-sat-lama-2011 {domain_file_path} {problem_file_path} >/dev/null 2>&1')
            plan_file_name = "sas_plan.1"
            with open(plan_file_name) as f:
                plan = [eval(line.replace('\n','').replace(' ','\', \'').replace('(','(\'').replace(')','\')')) for line in f.readlines() if 'unit cost' not in line]
            locations = set()
            for plan_tup in plan:
                for lexicon in plan_tup[1:]:
                    if "loc" in lexicon:
                        locations.add(lexicon)
            for loc in locations:
                loc_index = int(loc[len("loc"):])
                if self._granularity % 2 == 0:
                    square = int(np.sqrt(2 ** self._granularity))
                    rows, cols = square, square
                else:
                    square = int(np.sqrt(2 ** (self._granularity - 1)))
                    rows, cols = square, int((2 ** self._granularity) / square)
                loc_x, loc_y = loc_index // cols, loc_index % cols
                xmin, ymin = self._xy_bounds[:, 0]
                xmax, ymax = self._xy_bounds[:, 1]
                x_lower_bound = xmin + (xmax - xmin) / rows * loc_x
                x_upper_bound = xmin + (xmax - xmin) / rows * (loc_x + 1)
                y_lower_bound = ymin + (ymax - ymin) / cols * loc_y
                y_upper_bound = ymin + (ymax - ymin) / cols * (loc_y + 1)
                x_mid, y_mid = (x_lower_bound + x_upper_bound) / 2, (y_lower_bound + y_upper_bound) / 2
                pos = np.array([x_mid, y_mid, 1.0])
                self._subgoal_urdf_id.append(
                    self.robot.pb_client.load_geom(
                        "sphere", size=0.04, mass=0, base_pos=pos, rgba=[0, 0.8, 0.8, 0.8]
                    )
                )

        # Remove collision checking between the robot and the subgoal balls, as well as
        # between the box and the subgoal balls.
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            for sg in self._subgoal_urdf_id:
                self.robot.pb_client.setCollisionFilterPair(self.robot.arm.robot_id, sg, i, -1, enableCollision=0)
        for sg in self._subgoal_urdf_id:
            self.robot.pb_client.setCollisionFilterPair(self._box_id, sg, -1, -1, enableCollision=0)

    def reset(self):
        self.robot.arm.set_jpos(self._arm_reset_pos, ignore_physics=True)
        self.robot.pb_client.reset_body(self._box_id, base_pos=self._box_pos)
        self._t = 0
        self.max_plan_step_reached = 0
        self._ref_ee_pos = self.robot.arm.get_ee_pose()[0]
        self._ref_ee_ori = self.robot.arm.get_ee_pose()[1]
        return self._get_obs()

    def step(self, action):
        previous_state = self._get_obs()
        collision = self._apply_action(action)
        self._t += 1
        state = self._get_obs()
        done = self._t >= self._max_episode_length
        reward, info = self._get_reward(state=state, action=action, previous_state=previous_state)
        info["collision"] = collision
        return state, reward, done, info

    def _get_reward(self, state, action, previous_state):
        object_pos = state[2:4]
        dist_to_goal = np.linalg.norm(object_pos - self._goal_pos[:2])
        success = dist_to_goal < self._dist_threshold
        if self.reward_type == "sparse_handcrafted":
            reward = success
        elif self.reward_type == "dense_handcrafted":
            dist_to_obj = np.linalg.norm(state[:2] - object_pos)
            previous_dist_to_goal = np.linalg.norm(previous_state[2:4] - self._goal_pos[:2])
            if dist_to_obj < self._dist_threshold:
                reward = -2.0 * dist_to_obj ** 2 +  (-1.0 * dist_to_goal ** 3)
            else:
                reward = -1.0 * dist_to_goal ** 2 + (previous_dist_to_goal - dist_to_goal)
        else:
            reward = None
        info = {'success': success}
        return reward, info

    def _get_obs(self):
        gripper_pos = self.robot.arm.get_ee_pose()[0][:2]
        object_pos, object_quat = self.robot.pb_client.get_body_state(self._box_id)[:2]
        state = np.concatenate([gripper_pos, object_pos[:2]])
        return state

    def _apply_action(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 2:
            raise ValueError("Action should be [d_x, d_y].")
        # we set dz=0
        action = np.append(action, 0)
        pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
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

        return False

    def render(self, mode, **kwargs):
        robot_base = self.robot.arm.robot_base_pos
        self.robot.cam.setup_camera(
            focus_pt=robot_base, dist=2, yaw=85, pitch=-20, roll=0
        )
        rgb, _ = self.robot.cam.get_images(get_rgb=True, get_depth=False)
        return rgb

    def get_success(self, env, state):
        dist_to_goal = np.linalg.norm(state[0][2:4] - env._goal_pos[:2])
        return dist_to_goal < env._dist_threshold


module_name = __name__

env_name = "URPusher-v1"
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f"{module_name}:URRobotPusherGym",
)