#!/bin/bash
# echo "Running experiment... outputing here > reach_ppo_multi_subgoal.txt"
# python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal &> reach_ppo_multi_subgoal.txt

# echo "Running experiment... outputing here > reach_ppo_grid_based.txt"
# python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type grid_based &> reach_ppo_grid_based.txt

# echo "Running experiment... outputing here > push_ppo_multi_subgoal.txt"
# python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type multi_subgoal &> push_ppo_multi_subgoal.txt

echo "Running experiment... outputing here > push_ppo_grid_based.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type grid_based --granularity 5 &> push_ppo_grid_based.txt
