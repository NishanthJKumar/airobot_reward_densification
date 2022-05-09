---
layout: page
title: Automatic Reward Densification
<!-- tagline: Easy reward design using classical planning -->
description: Minimal tutorial on making a simple website with GitHub Pages
---

A core challenge with scaling up Deep Reinforcement Learning (Deep RL) for use in robotic tasks of practical interest is the task specification problem, which typically manifests as the difficulty of reward design. In order to reduce the difficulty of reward function design in continuous robotics environments, we propose to develop a method that automatically densifies sparse, goal-based reward in robotic tasks such that the optimal policy is preserved by leveraging task plans. 

We hypothesize that for many robotic tasks, 
- while it is difficult for humans to specify a dense reward that cannot be hacked, it is easy to specify an abstract plannable model in PDDL that conveys information about the dynamics of the domain, and 
- that valid abstract plans within this model can be leveraged to automatically densify sparse reward via potential-based reward shaping sufficiently enough for state-of-the-art RL approaches to solve these tasks. 

We perform an extensive empirical evaluation of our system across different PDDL models with varying granularity, choices of potential function, choice of learning algorithm (PPO and SAC) and tasks.

#### [Report](assets/report.pdf) | [Presentation](https://docs.google.com/presentation/d/1M5sPGWkCsGoGVnfnY0i7reYVwLuUrs1Fx4Vm10iyals/edit?usp=sharing) | [Code](https://github.com/NishanthJKumar/airobot_reward_densification)

---

### Environments

#### Reaching - PPO 

<table class="wide">
    <colgroup>
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
    </colgroup>
    <tr>
        <td colspan="3">
            <p style="text-align:center"> Sparse Handcrafted </p>
            <center>
                <img src="assets/videos/reach_ppo_sparse_handcrafted.gif"/>
            </center>
        </td>
        <td colspan="3">
            <p style="text-align:center"> Dense Handcrafted </p>
            <center>
                <img src="assets/videos/reach_ppo_dense_handcrafted.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Single Subgoal </p>
            <center>
                <img src="assets/videos/reach_ppo_simple_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Multi Subgoal </p>
            <center>
                <img src="assets/videos/reach_ppo_simple_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Grid Based </p>
            <center>
                <img src="assets/videos/reach_ppo_simple_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/reach_ppo_plan_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/reach_ppo_plan_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Grid Based </p>
            <center>
                <img src="assets/videos/reach_ppo_plan_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/reach_ppo_dist_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/reach_ppo_dist_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Grid Based </p>
            <center>
                <img src="assets/videos/reach_ppo_dist_grid.gif"/>
            </center>
        </td>
    </tr>
</table>

---

#### Reaching - SAC

<table class="wide">
    <colgroup>
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
    </colgroup>
    <tr>
        <td colspan="3">
            <p style="text-align:center"> Sparse Handcrafted </p>
            <center>
                <img src="assets/videos/reach_sac_sparse_handcrafted.gif"/>
            </center>
        </td>
        <td colspan="3">
            <p style="text-align:center"> Dense Handcrafted </p>
            <center>
                <img src="assets/videos/reach_sac_dense_handcrafted.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Single Subgoal </p>
            <center>
                <img src="assets/videos/reach_sac_simple_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Multi Subgoal </p>
            <center>
                <img src="assets/videos/reach_sac_simple_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Grid Based </p>
            <center>
                <img src="assets/videos/reach_sac_simple_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/reach_sac_plan_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/reach_sac_plan_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Grid Based </p>
            <center>
                <img src="assets/videos/reach_sac_plan_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/reach_sac_dist_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/reach_sac_dist_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Grid Based </p>
            <center>
                <img src="assets/videos/reach_sac_dist_grid.gif"/>
            </center>
        </td>
    </tr>
</table>

---

#### Pushing - PPO

<table class="wide">
    <colgroup>
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
    </colgroup>
    <tr>
        <td colspan="3">
            <p style="text-align:center"> Sparse Handcrafted </p>
            <center>
                <img src="assets/videos/push_ppo_sparse_handcrafted.gif"/>
            </center>
        </td>
        <td colspan="3">
            <p style="text-align:center"> Dense Handcrafted </p>
            <center>
                <img src="assets/videos/push_ppo_dense_handcrafted.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Single Subgoal </p>
            <center>
                <img src="assets/videos/push_ppo_simple_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Multi Subgoal </p>
            <center>
                <img src="assets/videos/push_ppo_simple_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Grid Based </p>
            <center>
                <img src="assets/videos/push_ppo_simple_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/push_ppo_plan_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/push_ppo_plan_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Grid Based </p>
            <center>
                <img src="assets/videos/push_ppo_plan_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/push_ppo_dist_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/push_ppo_dist_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Grid Based </p>
            <center>
                <img src="assets/videos/push_ppo_dist_grid.gif"/>
            </center>
        </td>
    </tr>
</table>

---

##### Pushing - SAC

<table class="wide">
    <colgroup>
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
    </colgroup>
    <tr>
        <td colspan="3">
            <p style="text-align:center"> Sparse Handcrafted </p>
            <center>
                <img src="assets/videos/push_sac_sparse_handcrafted.gif"/>
            </center>
        </td>
        <td colspan="3">
            <p style="text-align:center"> Dense Handcrafted </p>
            <center>
                <img src="assets/videos/push_sac_dense_handcrafted.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Single Subgoal </p>
            <center>
                <img src="assets/videos/push_sac_simple_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Multi Subgoal </p>
            <center>
                <img src="assets/videos/push_sac_simple_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Simple - Grid Based </p>
            <center>
                <img src="assets/videos/push_sac_simple_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/push_sac_plan_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/push_sac_plan_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Plan-Based - Grid Based </p>
            <center>
                <img src="assets/videos/push_sac_plan_grid.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/push_sac_dist_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/push_sac_dist_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Grid Based </p>
            <center>
                <img src="assets/videos/push_sac_dist_grid.gif"/>
            </center>
        </td>
    </tr>
</table>

---

#### Maze-Reach - PPO

<table class="wide">
    <colgroup>
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
        <col width="16.5%">
    </colgroup>
    <tr>
        <td colspan="6">
            <p style="text-align:center"> Sparse Handcrafted </p>
            <center>
                <img src="assets/videos/maze_reach_ppo_sparse_handcrafted.gif"/>
            </center>
        </td>
    </tr>
    <tr>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Single Subgoal </p>
            <center>
                <img src="assets/videos/maze_reach_ppo_dist_single.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Multi Subgoal </p>
            <center>
                <img src="assets/videos/maze_reach_ppo_dist_multi.gif"/>
            </center>
        </td>
        <td colspan="2">
            <br>
            <p style="text-align:center"> Distance-Based - Grid Based </p>
            <center>
                <img src="assets/videos/maze_reach_ppo_dist_grid.gif"/>
            </center>
        </td>
    </tr>
</table>

---
