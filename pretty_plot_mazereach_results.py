import pickle
import matplotlib.pyplot as plt
import numpy as np

def create_renamed_dict(input_dict):
    renaming = {'sparse_handcrafted': 'sparse_hc',
                'pddl_grid_based_distdrs': 'pddl\ngrid_based\ndist_drs',
                "pddl_single_subgoal_distdrs": "pddl\nsingle\nsubgoal\ndist_drs",
                "pddl_multi_subgoal_distdrs": "pddl\nmulti\nsubgoal\ndist_drs",
                }
    
    renamed_dict = {}
    for key in input_dict:
        new_key = renaming[key]
        renamed_dict[new_key] = input_dict[key]

    return renamed_dict

def combine_dicts(dict0, dict1, dict2):
    combined_dict = {}
    for key0 in dict0.keys():
        if 'sparse_handcrafted' in key0:
            k = 'sparse_handcrafted'
        elif 'dense_handcrafted' in key0:
            k = 'dense_handcrafted'
        else:
            k = key0

        k1 = None
        k2 = None
        for key1 in dict1.keys():
            if k in key1:
                k1 = key1
                break
        for key2 in dict2.keys():
            if k in key2:
                k2 = key2
                break

        if k1 is not None and k2 is not None:
            combined_dict[k] = [dict0[key0], dict1[k1], dict2[k2]]
        else:
            raise ValueError(f"Could not find key {k} in dicts")
    return combined_dict

with open('mazereach_results_seed0.pickle', 'rb') as handle:
    results_dict0 = pickle.load(handle)

with open('mazereach_results_seed1.pickle', 'rb') as handle:
    results_dict1 = pickle.load(handle)

with open('mazereach_results_seed3.pickle', 'rb') as handle:
    results_dict3 = pickle.load(handle)

# Combine all the dictionaries into one composite dictionary.
reach_ppo_maze_combined = combine_dicts(results_dict0, results_dict1, results_dict3)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

reach_ppo_maze_renamed = create_renamed_dict(reach_ppo_maze_combined)

key_ordering = ['sparse_hc', "pddl\nsingle\nsubgoal\ndist_drs", 
"pddl\nmulti\nsubgoal\ndist_drs", 'pddl\ngrid_based\ndist_drs']

plt.figure(0)
plt.bar(key_ordering, [np.array(reach_ppo_maze_renamed[k]).mean() for k in key_ordering], yerr=[np.array(reach_ppo_maze_renamed[k]).std() for k in key_ordering], color='tab:cyan')
plt.title('PPO Performance on Maze Reaching Task', fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Distance to Goal', fontsize=22)

plt.show()
