# from reaching_task import read_tf_log, plot_curves
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
#### 
# 
# TODO: plot return and success rate curves
steps_dense_reach, returns_dense_reach, success_rate_dense_reach = read_tf_log('data/URReacher-v1_dense_handcrafted_grid_based_ppo_200000_100_100_5')
steps_sparse_reach, returns_sparse_reach, success_rate_sparse_reach = read_tf_log('data/URReacher-v1_sparse_handcrafted_grid_based_ppo_200000_200_100_5')
plot_success_rate_dict = {'Dense': [steps_dense_reach, success_rate_dense_reach], 'Sparse': [steps_sparse_reach, success_rate_sparse_reach]}
plot_curves(plot_success_rate_dict, 'Reaching Success Rate')
plt.show()

steps_dense_push, returns_dense_push, success_rate_dense_push = read_tf_log('data/URPusher-v1_dense_handcrafted_grid_based_ppo_200000_100_100_5')
steps_sparse_push, returns_sparse_push, success_rate_sparse_push = read_tf_log('data/URPusher-v1_sparse_handcrafted_single_subgoal_ppo_200000_200_100_5')
plot_success_rate_dict = {'Dense': [steps_dense_push, success_rate_dense_push], 'Sparse': [steps_sparse_push, success_rate_sparse_push]}
plot_curves(plot_success_rate_dict, 'Pushing Success Rate')
plt.show()

steps_dist_single_reach, returns_dist_single_reach, success_rate_dist_single_reach = read_tf_log('data/URReacher-v1_pddl_single_subgoal_ppo_200000_200_100_5_dist')
steps_dist_multi_reach, returns_dist_multi_reach, success_rate_dist_multi_reach = read_tf_log('data/URReacher-v1_pddl_multi_subgoal_ppo_200000_200_100_5_dist')
steps_dist_grid_reach, returns_dist_grid_reach, success_rate_dist_grid_reach = read_tf_log('data/URReacher-v1_pddl_grid_based_ppo_200000_200_100_5_dist')
plot_success_rate_dict = {'Single': [steps_dist_single_reach, success_rate_dist_single_reach], 'Multi': [steps_dist_multi_reach, success_rate_dist_multi_reach], 'Grid': [steps_dist_grid_reach, success_rate_dist_grid_reach]}
plot_curves(plot_success_rate_dict, 'Reaching Success Rate')
plt.show()

steps_dist_single_push, returns_dist_single_push, success_rate_dist_single_push = read_tf_log('data/URPusher-v1_pddl_single_subgoal_ppo_200000_200_100_5_dist')
steps_dist_multi_push, returns_dist_multi_push, success_rate_dist_multi_push = read_tf_log('data/URPusher-v1_pddl_multi_subgoal_ppo_200000_200_100_5_dist')
steps_dist_grid_push, returns_dist_grid_push, success_rate_dist_grid_push = read_tf_log('data/URPusher-v1_pddl_grid_based_ppo_200000_200_100_5_dist')
plot_success_rate_dict = {'Single': [steps_dist_single_push, success_rate_dist_single_push], 'Multi': [steps_dist_multi_push, success_rate_dist_multi_push], 'Grid': [steps_dist_grid_push, success_rate_dist_grid_push]}
plot_curves(plot_success_rate_dict, 'Pushing Success Rate')
plt.show()

steps_pddl_single_reach, returns_pddl_single_reach, success_rate_pddl_single_reach = read_tf_log('data/URReacher-v1_pddl_single_subgoal_ppo_200000_200_100_5')
steps_pddl_multi_reach, returns_pddl_multi_reach, success_rate_pddl_multi_reach = read_tf_log('data/URReacher-v1_pddl_multi_subgoal_ppo_200000_200_100_5')
steps_pddl_grid_reach, returns_pddl_grid_reach, success_rate_pddl_grid_reach = read_tf_log('data/URReacher-v1_pddl_grid_based_ppo_200000_200_100_5')
plot_success_rate_dict = {'Single': [steps_pddl_single_reach, success_rate_pddl_single_reach], 'Multi': [steps_pddl_multi_reach, success_rate_pddl_multi_reach], 'Grid': [steps_pddl_grid_reach, success_rate_pddl_grid_reach]}
plot_curves(plot_success_rate_dict, 'Reaching Success Rate')
plt.show()

steps_pddl_single_push, returns_pddl_single_push, success_rate_pddl_single_push = read_tf_log('data/URPusher-v1_pddl_single_subgoal_ppo_200000_200_100_5')
steps_pddl_multi_push, returns_pddl_multi_push, success_rate_pddl_multi_push = read_tf_log('data/URPusher-v1_pddl_multi_subgoal_ppo_200000_200_100_5')
steps_pddl_grid_push, returns_pddl_grid_push, success_rate_pddl_grid_push = read_tf_log('data/URPusher-v1_pddl_grid_based_ppo_200000_200_100_5')
plot_success_rate_dict = {'Single': [steps_pddl_single_push, success_rate_pddl_single_push], 'Multi': [steps_pddl_multi_push, success_rate_pddl_multi_push], 'Grid': [steps_pddl_grid_push, success_rate_pddl_grid_push]}
plot_curves(plot_success_rate_dict, 'Pushing Success Rate')
plt.show()

# steps, returns, success_rate = read_tf_log('data/URReacher-v1_dense_handcrafted_grid_based_ppo_200000_100_100_5')
# plot_rewards_dict = {'returns': [steps, returns]}
# plot_curves(plot_rewards_dict, 'Sparse Reward Returns')
# plt.show()
# plot_success_rate_dict = {'success_rate': [steps, success_rate]}
# plot_curves(plot_success_rate_dict, 'Sparse Reward Success Rate')
# plt.show()