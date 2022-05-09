import pickle
import matplotlib.pyplot as plt
import numpy as np

def create_renamed_dict(input_dict):
    renaming = {'sparse_handcrafted': 'sparse_hc',
                'dense_handcrafted': 'dense_hc',
                'pddl_grid_based_basicdrs': 'pddl\ngrid_based\nbasic_drs',
                'pddl_multi_subgoal_': 'pddl\nmulti\nsubgoal',
                'pddl_grid_based_distdrs': 'pddl\ngrid_based\ndist_drs',
                "pddl_grid_based_":'pddl\ngrid_based',
                "pddl_multi_subgoal_basicdrs": "pddl\nmulti\nsubgoal\nbasic_drs",
                "pddl_single_subgoal_basicdrs": "pddl\nsingle\nsubgoal\nbasic_drs",
                "pddl_single_subgoal_distdrs": "pddl\nsingle\nsubgoal\ndist_drs",
                "pddl_multi_subgoal_distdrs": "pddl\nmulti\nsubgoal\ndist_drs",
                "pddl_single_subgoal_": "pddl\nsingle\nsubgoal",
                'sparse_handcrafted_single_subgoal_': "sparse_hc"}
    
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

with open('results_seed0.pickle', 'rb') as handle:
    results_list0 = pickle.load(handle)

with open('results_seed1.pickle', 'rb') as handle:
    results_list1= pickle.load(handle)

with open('results_seed3.pickle', 'rb') as handle:
    results_list3 = pickle.load(handle)

reach_ppo_results0, reach_sac_results0, push_ppo_results0, push_sac_results0 = results_list0
reach_ppo_results1, reach_sac_results1, push_ppo_results1, push_sac_results1 = results_list1
reach_ppo_results3, reach_sac_results3, push_ppo_results3, push_sac_results3 = results_list3

# Combine all the dictionaries into one composite dictionary.
reach_ppo_combined = combine_dicts(reach_ppo_results0, reach_ppo_results1, reach_ppo_results3)
reach_sac_combined = combine_dicts(reach_sac_results0, reach_sac_results1, reach_sac_results3)
push_ppo_combined = combine_dicts(push_ppo_results0, push_ppo_results1, push_ppo_results3)
push_sac_combined = combine_dicts(push_sac_results0, push_sac_results1, push_sac_results3)

reach_ppo_renamed = create_renamed_dict(reach_ppo_combined)
reach_sac_renamed = create_renamed_dict(reach_sac_combined)
push_ppo_renamed = create_renamed_dict(push_ppo_combined)
push_sac_renamed = create_renamed_dict(push_sac_combined)

key_ordering = ['sparse_hc', 'dense_hc', 
"pddl\nsingle\nsubgoal", 'pddl\nmulti\nsubgoal', 'pddl\ngrid_based', 
"pddl\nsingle\nsubgoal\nbasic_drs", "pddl\nmulti\nsubgoal\nbasic_drs", 
'pddl\ngrid_based\nbasic_drs', "pddl\nsingle\nsubgoal\ndist_drs", 
"pddl\nmulti\nsubgoal\ndist_drs", 'pddl\ngrid_based\ndist_drs']

width = 0.4

plt.figure(0)
x_axis = np.arange(len(key_ordering))
plt.bar(x_axis - width/2, [np.array(reach_ppo_renamed[k]).mean() for k in key_ordering], yerr=[np.array(reach_ppo_renamed[k]).std() for k in key_ordering], color='tab:blue', label='Reaching', width=width)
plt.bar(x_axis + width/2, [np.array(push_ppo_renamed[k]).mean() for k in key_ordering], yerr=[np.array(push_ppo_renamed[k]).std() for k in key_ordering], color='tab:orange', label='Pushing', width=width)
plt.title('PPO Performance on Reaching and Pushing Tasks', fontsize=22)
plt.xticks(x_axis, key_ordering, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.ylabel('Distance to Goal', fontsize=18)

plt.figure(1)
x_axis = np.arange(len(key_ordering))
plt.bar(x_axis - width/2, [np.array(reach_sac_renamed[k]).mean() for k in key_ordering], yerr=[np.array(reach_ppo_renamed[k]).std() for k in key_ordering], color='tab:blue', label='Reaching', width=width)
plt.bar(x_axis + width/2, [np.array(push_sac_renamed[k]).mean() for k in key_ordering], yerr=[np.array(push_ppo_renamed[k]).std() for k in key_ordering], color='tab:orange', label='Pushing', width=width)
plt.title('SAC Performance on Reaching and Pushing Tasks', fontsize=22)
plt.xticks(x_axis, key_ordering, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.ylabel('Distance to Goal', fontsize=18)

plt.show()
