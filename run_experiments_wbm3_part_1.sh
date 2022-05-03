#!/bin/bash
# (Reach and Push) x (SAC and PPO) x (Dense, Sparse, Reward_Shaping, and DNS_dist)

# Run Baseline Experiments
echo "Running experiment... outputing here > push_sac_dense.txt"
python main.py --domain push --algorithm sac --reward_type dense_handcrafted  --pddl_type single_subgoal  --path_to_fd /home/wbm3/Documents/GitHub/downward &> push_sac_dense.txt

echo "Running experiment... outputing here > push_ppo_dense.txt"
python main.py --domain push --algorithm ppo --reward_type dense_handcrafted  --pddl_type single_subgoal  --path_to_fd /home/wbm3/Documents/GitHub/downward &> push_ppo_dense.txt

echo "Running experiment... outputing here > push_sac_sparse.txt"
python main.py --domain push --algorithm sac --reward_type sparse_handcrafted  --pddl_type single_subgoal  --path_to_fd /home/wbm3/Documents/GitHub/downward &> push_sac_sparse.txt

echo "Running experiment... outputing here > push_ppo_sparse.txt"
python main.py --domain push --algorithm ppo --reward_type sparse_handcrafted  --pddl_type single_subgoal  --path_to_fd /home/wbm3/Documents/GitHub/downward &> push_ppo_sparse.txt
