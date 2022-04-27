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
steps, returns, success_rate = read_tf_log('data/ob_True0_gran5')
plot_rewards_dict = {'returns': [steps, returns]}
plot_curves(plot_rewards_dict, 'Sparse Reward Returns')
plt.show()
plot_success_rate_dict = {'success_rate': [steps, success_rate]}
plot_curves(plot_success_rate_dict, 'Sparse Reward Success Rate')
plt.show()