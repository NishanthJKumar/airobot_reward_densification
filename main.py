from easyrl.agents.ppo_agent import PPOAgent
from easyrl.configs import cfg, set_config
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env
from shaped_reward_episodic_runner import ShapedRewardEpisodicRunner
from torch import nn
import torch
from pathlib import Path
import pprint
# NOTE: Needed for env.registery to go through
import envs.pushing_env.pushing_task
import envs.reaching_env.reaching_task
from utils import play_video, GroundingUtils
import gym
from envs.pushing_env.single_subgoal.single_subgoal import PushingSingleSubgoalClassfiers
from envs.pushing_env.multiple_subgoals.multiple_subgoals import PushingMultipleSubgoalClassfiers
from envs.pushing_env.grid_based.grid_based import PushingGridBasedClassifiers
from envs.reaching_env.multiple_subgoals.multiple_subgoals import MultipleSubgoalsClassfiers
from envs.reaching_env.single_subgoal.single_subgoal import SingleSubgoalClassfiers
from envs.reaching_env.grid_based.grid_based import GridBasedClassifiers

def train_ppo(
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
    runner = ShapedRewardEpisodicRunner(g_utils=grounding_utils, agent=agent, env=env)
    engine = PPOEngine(agent=agent, runner=runner)
    if cfg.alg.eval:
        stat_info, _ = engine.eval(
            render=False, save_eval_traj=True, eval_num=1, sleep_time=0.0
        )
        pprint.pprint(stat_info)
        play_video(cfg.alg.save_dir+"/seed_"+str(cfg.alg.seed))
    else:
        engine.train()
        stat_info, _ = engine.eval(
            render=False, save_eval_traj=True, eval_num=1, sleep_time=0.0
        )
        pprint.pprint(stat_info)
        play_video(cfg.alg.save_dir+"/seed_"+str(cfg.alg.seed))
    
    return cfg.alg.save_dir

# Structure of the remainder of this file:
# 1. Take in input on whether we're training or evaling
# 2. Take in input on environment
# 3. Take in input on the kind of shaping that we want.
# 4. Take in optional input on hyperparams for training/eval.
# 5. Run the appropriate function (training or evaling) in the 
# appropriate environment.

classifiers = PushingGridBasedClassifiers()
domain_file_path, problem_file_path = classifiers.get_path_to_domain_and_problem_files()
path_to_fd_folder = '/home/njk/Documents/GitHub/downward'

# call train_ppo, just set the argument flag properly
push_exp = True
granularity = 3
with_obstacle=False
env_name="URPusher-v1" if push_exp else "URReacher-v1"
max_steps=150000

set_config("ppo")
cfg.alg.seed = 0
cfg.alg.num_envs = 1
cfg.alg.max_steps = max_steps
cfg.alg.deque_size = 20
cfg.alg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.alg.eval = False
cfg.alg.env_name = env_name
cfg.alg.dynamic_reward_shaping = False
cfg.alg.save_dir = Path.cwd().absolute().joinpath("data").as_posix()
cfg.alg.save_dir += "/" + f"{env_name}"
if push_exp:
    cfg.alg.save_dir += "_push"
cfg.alg.save_dir += f"ob_{str(with_obstacle)}"
cfg.alg.save_dir += str(cfg.alg.seed)
cfg.alg.episode_steps = 25
cfg.alg.eval_interval = 50
setattr(cfg.alg, "diff_cfg", dict(save_dir=cfg.alg.save_dir))

print(f"====================================")
print(f"      Device:{cfg.alg.device}")
print(f"      Total number of steps:{cfg.alg.max_steps}")
print(f"====================================")

set_random_seed(cfg.alg.seed)
env_kwargs = (
    dict(
        with_obstacle=with_obstacle,
        granularity=granularity,
    )
    if not push_exp
    else dict()
)
env_kwargs.update(dict(max_episode_length = cfg.alg.episode_steps))
env = make_vec_env(
    cfg.alg.env_name, cfg.alg.num_envs, seed=cfg.alg.seed, env_kwargs=env_kwargs
)

grounding_utils = GroundingUtils(domain_file_path, problem_file_path, env, classifiers, path_to_fd_folder, env.envs[0].get_success)
save_dir = train_ppo(
    cfg=cfg,
    env_name="URPusher-v1" if push_exp else "URReacher-v1",
    grounding_utils=grounding_utils,
)

#### TODO: plot return and success rate curves
# steps, returns, success_rate = read_tf_log(save_dir)
# data_dict = {}
# data_dict["return"] = [steps, returns]
# plot_curves(data_dict, "Reaching Task without Obstacles - Sparse Reward")
# data_dict = {}
# data_dict["success_rate"] = [steps, success_rate]
# plot_curves(data_dict, "Reaching Task without Obstacles - Sparse Reward")
