# Scheduled Policy Optimization for Natural Language Communication with Intelligent Agents

## Models and Algorithms
See files under `walk_the_blocks/BlockWorldRoboticAgent/srcs/`

* `learn_by_ppo.py` 
   run this file for training, you can change the schedule mechanism in the function `ppo_update()`, these are the options:
   * do imitation every 50
   * do imitation based on rules
   * imitation 1 epoch and then RL 1 epoch

   example:
   `python learn_by_ppo.py -lr 0.0001 -max_epochs 2 -entropy_coef 0.05`
* `policy_model.py`
   the network achitecture and loss functions:
   * PPO Loss
   * Supervised Loss
   * Advantage Actor-Critic Loss

## Instructions
For the usage of the Block-world environment, please refer to [https://github.com/clic-lab/blocks](https://github.com/clic-lab/blocks)

### Train the RL agents
* S-REIN
   * 
   
## If you use our code in your own research, please cite the following paper
`
@article{xiong2018scheduled,
  title={Scheduled Policy Optimization for Natural Language Communication with Intelligent Agents},
  author={Xiong, Wenhan and Guo, Xiaoxiao and Yu, Mo and Chang, Shiyu and Zhou, Bowen and Wang, William Yang},
  journal={arXiv preprint arXiv:1806.06187},
  year={2018}
}
`
