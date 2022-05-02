import gym
import numpy as np
from gym import spaces
from gym.envs.registration import registry, register

from airobot import Robot
from airobot.utils.common import ang_in_mpi_ppi
from airobot.utils.common import clamp
from airobot.utils.common import euler2quat
from airobot.utils.common import quat_multiply
from airobot.utils.common import rotvec2quat


class URRobotPickerGym(gym.Env):
    def __init__(self, action_repeat=10, gui=False, max_episode_length=25, dist_threshold = 0.05):
        self._action_repeat = action_repeat
        self._max_episode_length = max_episode_length
        self._dist_threshold = dist_threshold
        self.robot = Robot('ur5e_2f140',
                           pb_cfg={'gui': gui,
                                   'realtime': False})
        self.ee_ori = [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0]
        self._action_bound = 1.0
        self._ee_pos_scale = 0.02
        self._ee_ori_scale = np.pi / 36.0
        self._xy_bounds = np.array([[0.23, 0.78], # [xmin, xmax]
                                   [-0.35, 0.3]]) # [ymin, ymax]
        # create a ball at the goal location
        self._goal_pos = np.array([0.65, 0.25, 1.1])
        self._goal_urdf_id = self.robot.pb_client.load_geom(
            "sphere", size=0.04, mass=0, base_pos=self._goal_pos, rgba=[1, 0, 0, 0.8]
        )

        # disable the collision checking between the robot and the ball at the goal location
        for i in range(self.robot.pb_client.getNumJoints(self.robot.arm.robot_id)):
            self.robot.pb_client.setCollisionFilterPair(
                self.robot.arm.robot_id, self._goal_urdf_id, i, -1, enableCollision=0
            )
        self._box_pos = np.array([0.35, -0.1, 0.996])
        self._box_id = self.robot.pb_client.load_geom('box', size=0.05, mass=1,
                                                     base_pos=[0.5, 0.12, 1.0],
                                                     rgba=[1, 0, 0, 1])

        self.robot.pb_client.changeDynamics(self._box_id, -1, lateralFriction=0.9)
        self.robot.pb_client.setCollisionFilterPair(self._box_id, self._goal_urdf_id, -1, -1, enableCollision=0)

        self._action_high = np.array([self._action_bound] * 5)
        self.action_space = spaces.Box(low=-self._action_high,
                                       high=self._action_high,
                                       dtype=np.float32)
        state_low = np.full(len(self._get_obs()), -float('inf'))
        state_high = np.full(len(self._get_obs()), float('inf'))
        self.observation_space = spaces.Box(state_low,
                                            state_high,
                                            dtype=np.float32)
        self.reset()
        # add the dummy subgoal locations
        self._subgoal_urdf_id = []
        self._subgoal0_pos = self._ref_ee_pos
        self._subgoal1_pos = self._box_pos + np.array([0.0, 0.0, 0.25])
        self._subgoal_urdf_id.append(
            self.robot.pb_client.load_geom(
                "sphere", size=0.04, mass=0, base_pos=self._subgoal1_pos, rgba=[0, 0.8, 0.8, 0.8]
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
        self.robot.arm.reset()
        self.robot.arm.go_home(ignore_physics=True)
        ori = euler2quat([0, 0, np.pi / 2])
        self.table_id = self.robot.pb_client.load_urdf('table/table.urdf',
                                                       [.5, 0, 0.4],
                                                       ori,
                                                       scaling=0.9)
        self._ref_ee_pos = self.robot.arm.get_ee_pose()[0]
        self.ref_ee_ori = self.robot.arm.get_ee_pose()[1]
        self.gripper_ori = 0
        self._t = 0
        self.robot.arm.eetool.set_jpos(0.0)
        self._box_id = self.robot.pb_client.load_geom('box', size=0.05, mass=1,
                                                     base_pos=[0.5, 0.12, 1.0],
                                                     rgba=[1, 0, 0, 1])
        for step in range(self._action_repeat * 2):
            self.robot.pb_client.stepSimulation()
        return self._get_obs()

    def step(self, action):
        self.apply_action(action)
        self._t += 1
        state = self._get_obs()
        reward, info = self._get_reward(state)
        done = done = self._t >= self._max_episode_length
        return state, reward, done, info

    def _get_obs(self):
        # The observation is the robot's current 3D EE pos
        # and the 3D pos of the block.
        ee_pos = self.robot.arm.get_ee_pose()[0]
        gripper_open_pos = np.array([self.robot.arm.eetool.get_jpos()])
        object_pos, object_quat = self.robot.pb_client.get_body_state(self._box_id)[:2]
        state = np.concatenate([ee_pos, gripper_open_pos, object_pos])
        return state

    def _get_reward(self, state):
        object_pos = state[4:]
        dist_to_goal = np.linalg.norm(object_pos - self._goal_pos)
        success = dist_to_goal < self._dist_threshold
        reward = None
        info = {'success': success}
        return reward, info

    def apply_action(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action).flatten()
        if action.size != 5:
            raise ValueError('Action should be [d_x, d_y, d_z, '
                             'd_angle, open/close gripper].')
        pos, quat, rot_mat, euler = self.robot.arm.get_ee_pose()
        pos += action[:3] * self._ee_pos_scale

        self.gripper_ori += action[3] * self._ee_ori_scale
        self.gripper_ori = ang_in_mpi_ppi(self.gripper_ori)
        rot_vec = np.array([0, 0, 1]) * self.gripper_ori
        rot_quat = rotvec2quat(rot_vec)
        ee_ori = quat_multiply(self.ref_ee_ori, rot_quat)
        jnt_pos = self.robot.arm.compute_ik(pos, ori=ee_ori)
        gripper_ang = self._scale_gripper_angle(action[4])

        for step in range(self._action_repeat):
            self.robot.arm.set_jpos(jnt_pos)
            self.robot.arm.eetool.set_jpos(gripper_ang)
            self.robot.pb_client.stepSimulation()

    def _scale_gripper_angle(self, command):
        """
        Convert the command in [-1, 1] to the actual gripper angle.
        command = -1 means open the gripper.
        command = 1 means close the gripper.

        Args:
            command (float): a value between -1 and 1.
                -1 means open the gripper.
                1 means close the gripper.

        Returns:
            float: the actual gripper angle
            corresponding to the command.
        """
        command = clamp(command, -1.0, 1.0)
        close_ang = self.robot.arm.eetool.gripper_close_angle
        open_ang = self.robot.arm.eetool.gripper_open_angle
        cmd_ang = (command + 1) / 2.0 * (close_ang - open_ang) + open_ang
        return cmd_ang

    def render(self, **kwargs):
        robot_base = self.robot.arm.robot_base_pos
        self.robot.cam.setup_camera(focus_pt=robot_base,
                                    dist=3,
                                    yaw=55,
                                    pitch=-30,
                                    roll=0)

        rgb, _ = self.robot.cam.get_images(get_rgb=True,
                                           get_depth=False)
        return rgb

    def get_success(self, env, state):
        object_pos = state[0][4:]
        dist_to_goal = np.linalg.norm(object_pos - self._goal_pos)
        return dist_to_goal < env._dist_threshold


module_name = __name__

env_name = "URPicker-v1"
if env_name in registry.env_specs:
    del registry.env_specs[env_name]
register(
    id=env_name,
    entry_point=f"{module_name}:URRobotPickerGym",
)

# def main():
#     """
#     This function shows an example of block stacking.
#     """
#     np.set_printoptions(precision=4, suppress=True)
#     robot = Robot('ur5e_2f140')
#     success = robot.arm.go_home()
#     ori = euler2quat([0, 0, np.pi / 2])
#     robot.pb_client.load_urdf('table/table.urdf',
#                               [.5, 0, 0.4],
#                               ori,
#                               scaling=0.9)
#     box_size = 0.05
#     box_id1 = robot.pb_client.load_geom('box', size=box_size,
#                                         mass=.1,
#                                         base_pos=[.5, 0.12, 1.0],
#                                         rgba=[1, 0, 0, 1])
#     box_id2 = robot.pb_client.load_geom('box',
#                                         size=box_size,
#                                         mass=.1,
#                                         base_pos=[0.3, 0.12, 1.0],
#                                         rgba=[0, 0, 1, 1])
#     robot.arm.eetool.open()
#     obj_pos = robot.pb_client.get_body_state(box_id1)[0]
#     move_dir = obj_pos - robot.arm.get_ee_pose()[0]
#     move_dir[2] = 0
#     eef_step = 0.025
#     robot.arm.move_ee_xyz(move_dir, eef_step=eef_step)
#     move_dir = np.zeros(3)
#     move_dir[2] = obj_pos[2] - robot.arm.get_ee_pose()[0][2]
#     robot.arm.move_ee_xyz(move_dir, eef_step=eef_step)
#     robot.arm.eetool.close(wait=False)
#     import ipdb; ipdb.set_trace()
#     robot.arm.move_ee_xyz([0, 0, 0.3], eef_step=eef_step)

#     obj_pos = robot.pb_client.get_body_state(box_id2)[0]
#     move_dir = obj_pos - robot.arm.get_ee_pose()[0]
#     move_dir[2] = 0
#     robot.arm.move_ee_xyz(move_dir, eef_step=eef_step)
#     move_dir = obj_pos - robot.arm.get_ee_pose()[0]
#     move_dir[2] += box_size * 2
#     robot.arm.move_ee_xyz(move_dir, eef_step=eef_step)
#     robot.arm.eetool.open()
#     time.sleep(10)


# if __name__ == '__main__':
#     main()