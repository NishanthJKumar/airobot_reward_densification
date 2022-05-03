#!/bin/bash
# (Reach and Push) x (SAC and PPO) x (3 pddls)

# Run Reaching Experiments
echo "Running experiment... outputing here > reach_sac_single_subgoal.txt"
python main.py --domain "reach" --algorithm "sac" --reward_type "pddl" --pddl_type "single_subgoal" &> reach_sac_single_subgoal.txt

echo "Running experiment... outputing here > reach_sac_multi_subgoal.txt"
python main.py --domain "reach" --algorithm "sac" --reward_type "pddl" --pddl_type "multi_subgoal" &> reach_sac_multi_subgoal.txt

echo "Running experiment... outputing here > reach_sac_grid_based.txt"
python main.py --domain "reach" --algorithm "sac" --reward_type "pddl" --pddl_type "grid_based" &> reach_sac_grid_based.txt

echo "Running experiment... outputing here > reach_ppo_single_subgoal.txt"
python main.py --domain "reach" --algorithm "ppo" --reward_type "pddl" --pddl_type "single_subgoal" &> reach_ppo_single_subgoal.txt

echo "Running experiment... outputing here > reach_ppo_multi_subgoal.txt"
python main.py --domain "reach" --algorithm "ppo" --reward_type "pddl" --pddl_type "multi_subgoal" &> reach_ppo_multi_subgoal.txt

echo "Running experiment... outputing here > reach_ppo_grid_based.txt"
python main.py --domain "reach" --algorithm "ppo" --reward_type "pddl" --pddl_type "grid_based" &> reach_ppo_grid_based.txt

# Run Pushing Experiments
echo "Running experiment... outputing here > push_sac_single_subgoal.txt"
python main.py --domain "push" --algorithm "sac" --reward_type "pddl" --pddl_type "single_subgoal" &> push_sac_single_subgoal.txt

echo "Running experiment... outputing here > push_sac_multi_subgoal.txt"
python main.py --domain "push" --algorithm "sac" --reward_type "pddl" --pddl_type "multi_subgoal" &> push_sac_multi_subgoal.txt

echo "Running experiment... outputing here > push_sac_grid_based.txt"
python main.py --domain "push" --algorithm "sac" --reward_type "pddl" --pddl_type "grid_based" &> push_sac_grid_based.txt

echo "Running experiment... outputing here > push_ppo_single_subgoal.txt"
python main.py --domain "push" --algorithm "ppo" --reward_type "pddl" --pddl_type "single_subgoal" &> push_ppo_single_subgoal.txt

echo "Running experiment... outputing here > push_ppo_multi_subgoal.txt"
python main.py --domain "push" --algorithm "ppo" --reward_type "pddl" --pddl_type "multi_subgoal" &> push_ppo_multi_subgoal.txt

echo "Running experiment... outputing here > push_ppo_grid_based.txt"
python main.py --domain "push" --algorithm "ppo" --reward_type "pddl" --pddl_type "grid_based" &> push_ppo_grid_based.txt
