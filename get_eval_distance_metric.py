from easyrl.agents.ppo_agent import PPOAgent
from easyrl.agents.sac_agent import SACAgent
from easyrl.configs import cfg, set_config
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.engine.sac_engine import SACEngine
from easyrl.replays.circular_buffer import CyclicBuffer
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env
from shaped_reward_episodic_runner import ShapedRewardEpisodicRunner
from torch import nn
import argparse
import torch
from pathlib import Path
import pprint
import os
# NOTE: Needed for env.registery to go through
import envs.picking_env.picking_task
import envs.pushing_env.pushing_task
import envs.reaching_env.reaching_task
from utils import play_video, GroundingUtils
import gym
import numpy as np
from envs.picking_env.orig_blocksworld.single_subgoal import PickingSingleSubgoalClassfiers
from envs.picking_env.multiple_subgoals.multiple_subgoals import PickingMultipleSubgoalClassfiers
from envs.pushing_env.single_subgoal.single_subgoal import PushingSingleSubgoalClassfiers
from envs.pushing_env.multiple_subgoals.multiple_subgoals import PushingMultipleSubgoalClassfiers
from envs.pushing_env.grid_based.grid_based import PushingGridBasedClassifiers
from envs.reaching_env.multiple_subgoals.multiple_subgoals import ReachingMultipleSubgoalsClassfiers
from envs.reaching_env.single_subgoal.single_subgoal import ReachingSingleSubgoalClassfiers
from envs.reaching_env.grid_based.grid_based import ReachingGridBasedClassifiers

def eval_ppo(
    cfg=None,
    env_name="URPusher-v1",
    grounding_utils=None,
):

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
    runner = ShapedRewardEpisodicRunner(g_utils=grounding_utils, agent=agent, env=env, dynamic_reward_shaping=cfg.alg.dynamic_reward_shaping)
    engine = PPOEngine(agent=agent, runner=runner)
    agent.load_model()
    stat_info, raw_traj_info = engine.eval(
        render=False, save_eval_traj=True, eval_num=1, sleep_time=0.0
    )
    pprint.pprint(stat_info)

    return raw_traj_info

def eval_sac(
    cfg=None,
    env_name="URPusher-v1",
    grounding_utils=None,
):

    env.reset()
    ob_size = env.observation_space.shape[0]
    actor_body = MLP(
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
            tanh_on_dist=True,
            std_cond_in=True,
        )
    else:
        raise TypeError(f"Unknown action space type: {env.action_space}")

    q1_body = MLP(
        input_size=ob_size + act_size,
        hidden_sizes=[64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh,
    )

    q2_body = MLP(
        input_size=ob_size + act_size,
        hidden_sizes=[64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh,
    )

    q1 = ValueNet(q1_body)
    q2 = ValueNet(q2_body)
    memory = CyclicBuffer(capacity=cfg.alg.replay_size)
    agent = SACAgent(actor=actor, q1=q1, q2=q2, memory=memory, env=env)
    runner = ShapedRewardEpisodicRunner(g_utils=grounding_utils, agent=agent, env=env, dynamic_reward_shaping=cfg.alg.dynamic_reward_shaping)
    engine = SACEngine(agent=agent, runner=runner)
    agent.load_model()
    stat_info, traj_info = engine.eval(
        render=False, save_eval_traj=True, eval_num=1, sleep_time=0.0
    )
    pprint.pprint(stat_info)

    return traj_info


# Main code begins here; takes in particular arguments and urns the relevant experiment with the specified configuration.
parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--domain', choices=['reach', 'push', 'pick'], required=True, help='Name of env to run.')
# parser.add_argument('-rt', '--reward_type', choices=['sparse_handcrafted', "dense_handcrafted", 'pddl'], required=True, help='Type of reward to use.')
# parser.add_argument('-pt', '--pddl_type', choices=['single_subgoal', 'multi_subgoal', 'grid_based'], required=True, help='Type of classifier to use.')
# parser.add_argument('-al', '--algorithm', choices=['ppo', 'sac'], required=True, help='Choice of learning algorithm to use.')
# parser.add_argument('-ts', '--training_steps', type=int, default=200000, help='Number of steps to run training for.')
# parser.add_argument('-es', '--episode_steps', type=int, default=200, help='Max. number of steps in an episode.')
# parser.add_argument('-ei', '--eval_interval', type=int, default=100, help='Num. trajs after which to call eval.')
parser.add_argument('-fdp', '--path_to_fd', type=str, default="/home/njk/Documents/GitHub/downward", help='Full abs path to fd installation folder.')
# parser.add_argument('-se', '--seed', type=int, default=0, help='Random seed to use during training.')
# parser.add_argument('-g', '--granularity', type=int, default=5, help='Number of divisions to segment the working space of the arm. Total divisions is equal to 2^{input}.')
# parser.add_argument('-drs', '--dynamic_shaping', choices=['basic', 'dist'], nargs='?', help='DRS type to use.')
args = parser.parse_args()
args.seed = 0

all_data_folders = [f.path for f in os.scandir('data') if f.is_dir()]
# Loop thru all folders and populate the above lists.
for data_folder in all_data_folders:
    args_list = data_folder.split('/')[1].split('_')

    if args_list[0] == 'URReacher-v1':
        args.domain = 'reach'
    elif args_list[0] == 'URPusher-v1':
        args.domain = 'push'
    elif args_list[0] == 'URPicker-v1':
        args.domain = 'pick'

    if args_list[1] == 'pddl':
        args.reward_type = args_list[1]
        args.pddl_type = args_list[2] + '_' + args_list[3]
        args.algorithm = args_list[4]
        args.training_steps = int(args_list[5])
        args.episode_steps = int(args_list[6])
        args.eval_interval = int(args_list[7])
    else:
        args.reward_type = args_list[1] + "_" + args_list[2]
        args.pddl_type = args_list[3] + '_' + args_list[4]
        args.algorithm = args_list[5]
        args.training_steps = int(args_list[6])
        args.episode_steps = int(args_list[7])
        args.eval_interval = int(args_list[8])
    
    if args_list[-1] in ['basic', 'dist']:
        args.dynamic_shaping = args_list[-1]
    else:
        args.dynamic_shaping = None

    if args.dynamic_shaping is not None:
        args.granularity = int(args_list[-2])
    else:
        args.granularity = int(args_list[-1])

    env_kwargs = dict(reward_type = args.reward_type, gui = False)


    if args.domain == 'reach':
        env_name = "URReacher-v1"
        env_kwargs.update(dict(with_obstacle=True))
        if args.pddl_type == "single_subgoal":
            classifiers = ReachingSingleSubgoalClassfiers()
        elif args.pddl_type == "multi_subgoal":
            classifiers = ReachingMultipleSubgoalsClassfiers()
        elif args.pddl_type == "grid_based":
            env_kwargs.update(dict(granularity = args.granularity))
            classifiers = ReachingGridBasedClassifiers()
        else:
            raise ValueError(f"Unknown pddl type: {args.pddl_type}")
    elif args.domain == 'push':
        env_name = "URPusher-v1"
        if args.pddl_type == "single_subgoal":
            classifiers = PushingSingleSubgoalClassfiers()
        elif args.pddl_type == "multi_subgoal":
            classifiers = PushingMultipleSubgoalClassfiers()
        elif args.pddl_type == "grid_based":
            env_kwargs.update(dict(granularity = args.granularity))
            classifiers = PushingGridBasedClassifiers()
        else:
            raise ValueError(f"Unknown pddl type: {args.pddl_type}")
    elif args.domain == 'pick':
        env_name = "URPicker-v1"
        if args.pddl_type == "single_subgoal":
            classifiers = PickingSingleSubgoalClassfiers()
        elif args.pddl_type == "multi_subgoal":
            classifiers = PickingMultipleSubgoalClassfiers()
        else:
            raise ValueError(f"Unknown pddl type for picking env: {args.pddl_type}")
    else:
        raise ValueError(f"Unknown domain: {args.domain}")

    domain_file_path, problem_file_path = classifiers.get_path_to_domain_and_problem_files()

    set_config(args.algorithm)
    cfg.alg.seed = args.seed
    cfg.alg.num_envs = 1
    cfg.alg.epsilon = None
    cfg.alg.max_steps = args.training_steps
    cfg.alg.deque_size = 20
    cfg.alg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.alg.eval = True
    cfg.alg.test = True
    cfg.alg.resume = False
    cfg.alg.resume_step = None
    cfg.alg.env_name = env_name
    cfg.alg.dynamic_reward_shaping = args.dynamic_shaping
    cfg.alg.episode_steps = args.episode_steps
    cfg.alg.eval_interval = args.eval_interval
    cfg.alg.dynamic_reward_shaping = args.dynamic_shaping
    cfg.alg.pddl_type = args.pddl_type
    # Include all relevant variables in the name so that there are no folder
    # collisions.
    cfg.alg.save_dir = Path.cwd().absolute().joinpath("data").as_posix()
    cfg.alg.save_dir += "/" + f"{env_name}"
    cfg.alg.save_dir += "_" + args.reward_type + "_" + args.pddl_type + "_" + args.algorithm
    cfg.alg.save_dir += "_" + str(args.training_steps) + "_" + str(args.episode_steps) + "_" + str(args.eval_interval) + "_" + str(args.granularity)
    if args.dynamic_shaping is not None:
        cfg.alg.save_dir += "_" + args.dynamic_shaping
    setattr(cfg.alg, "diff_cfg", dict(save_dir=cfg.alg.save_dir))

    print(f"====================================")
    print(f"      Device:{cfg.alg.device}")
    print(f"      Total number of steps:{cfg.alg.max_steps}")
    print(f"====================================")

    set_random_seed(cfg.alg.seed)
    env_kwargs.update(dict(max_episode_length = 25))
    env = make_vec_env(
        cfg.alg.env_name, cfg.alg.num_envs, seed=cfg.alg.seed, env_kwargs=env_kwargs
    )

    grounding_utils = GroundingUtils(domain_file_path, problem_file_path, env, classifiers, args.path_to_fd, env.envs[0].get_success, cfg.alg.pddl_type)
    if args.algorithm == "ppo":
        traj_info = eval_ppo(
            cfg=cfg,
            env_name=env_name,
            grounding_utils=grounding_utils,
        )
    elif args.algorithm == "sac":
        traj_info = eval_sac(
            cfg=cfg,
            env_name=env_name,
            grounding_utils=grounding_utils,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")


    final_ob = traj_info['lst_step_info'][0]['true_next_ob']
    if args.domain == "reach":
        final_dist = np.linalg.norm(env.envs[0]._goal_pos[:2] - final_ob)
    elif args.domain == "push":
        final_dist = np.linalg.norm(env.envs[0]._goal_pos[:2] - final_ob[2:4])
    else:
        raise ValueError(f"Domain not yet implemented for eval: {args.domain}")

    print(f"{data_folder.split('/')[1]}, {cfg.alg.save_dir}: {final_dist}")

