#!/bin/bash
# (Reach and Push) x (SAC and PPO) x (Dense, Sparse, Reward_Shaping, and DNS_dist)

# Run Reward Shaping Experiments
echo "Running experiment... outputing here > reach_sac_dist.txt"
python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist --path_to_fd /home/wbm3/Documents/GitHub/downward &> reach_sac_dist_mg.txt

echo "Running experiment... outputing here > reach_ppo_dist.txt"
python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist --path_to_fd /home/wbm3/Documents/GitHub/downward &> reach_ppo_dist_mg.txt

echo "Running experiment... outputing here > push_sac_dist.txt"
python main.py --domain push --algorithm sac --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist --path_to_fd /home/wbm3/Documents/GitHub/downward &> push_sac_dist_mg.txt

echo "Running experiment... outputing here > push_ppo_dist.txt"
python main.py --domain push --algorithm ppo --reward_type pddl --pddl_type multi_subgoal --dynamic_shaping dist  --path_to_fd /home/wbm3/Documents/GitHub/downward &> push_ppo_dist_mg.txt

#echo "Running experiment... outputing here > reach_sac_dns.txt"
#python main.py --domain reach --algorithm sac --reward_type pddl --pddl_type multi_subgoal --path_to_fd /home/wbm3/Documents/GitHub/downward &> reach_sac_dns.txt

#echo "Running experiment... outputing here > reach_ppo_dns.txt"
#python main.py --domain reach --algorithm ppo --reward_type pddl --pddl_type multi_subgoal  --path_to_fd /home/wbm3/Documents/GitHub/downward &> reach_ppo_dns.txt