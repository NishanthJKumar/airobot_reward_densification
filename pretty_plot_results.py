import pickle
import matplotlib.pyplot as plt

def create_renamed_dict(input_dict):
    renaming = {'sparse_handcrafted_grid_based_': 'sparse_hc',
                'dense_handcrafted_grid_based_': 'dense_hc',
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



with open('results.pickle', 'rb') as handle:
    results_list = pickle.load(handle)

reach_ppo_results, reach_sac_results, push_ppo_results, push_sac_results = results_list

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lime']

reach_ppo_renamed = create_renamed_dict(reach_ppo_results)
reach_sac_renamed = create_renamed_dict(reach_sac_results)
push_ppo_renamed = create_renamed_dict(push_ppo_results)
push_sac_renamed = create_renamed_dict(push_sac_results)

key_ordering = ['sparse_hc', 'dense_hc', 
"pddl\nsingle\nsubgoal", 'pddl\nmulti\nsubgoal', 'pddl\ngrid_based', 
"pddl\nsingle\nsubgoal\nbasic_drs", "pddl\nmulti\nsubgoal\nbasic_drs", 
'pddl\ngrid_based\nbasic_drs', "pddl\nsingle\nsubgoal\ndist_drs", 
"pddl\nmulti\nsubgoal\ndist_drs", 'pddl\ngrid_based\ndist_drs']

plt.figure(0)
plt.bar(key_ordering, [reach_ppo_renamed[k] for k in key_ordering], color=colors)
plt.title('PPO Performance on Reaching Task')
plt.xlabel('Approach')
plt.ylabel('Distance to Goal')

plt.figure(1)
plt.bar(key_ordering, [reach_sac_renamed[k] for k in key_ordering], color=colors)
plt.title('SAC Performance on Reaching Task')
plt.xlabel('Approach')
plt.ylabel('Distance to Goal')

plt.figure(2)
plt.bar(key_ordering, [push_ppo_renamed[k] for k in key_ordering], color=colors)
plt.title('PPO Performance on Pushing Task')
plt.xlabel('Approach')
plt.ylabel('Distance to Goal')

plt.figure(3)
plt.bar(key_ordering, [push_sac_renamed[k] for k in key_ordering], color=colors)
plt.title('SAC Performance on Pushing Task')
plt.xlabel('Approach')
plt.ylabel('Distance to Goal')

plt.show()
