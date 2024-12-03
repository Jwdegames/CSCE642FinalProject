# CSCE642FinalProject
CSCE 642 Deep Reinforcement Learning Final Project - Multi-Agent vs Single-Agent Hierarchical Reinforcement Learning In A MOBA Game

This project pits a Multi-Agent Hierarchical Reinforcement Learning (MAHRL) architecture against Single-Agent Hierarchical Reinforcement Learning (SAHRL) architecture in Derk's Gym. One team is has three players each controlled by a separate MAHRL agent while another team has all three players controlled by a single SAHRL agent. 

Note: Game Environment is Dirk's Gym and belongs to Fredrik Norén, Mount Rouke. Find more info here: https://gym.derkgame.com/license \
A2C code is modified from CSCE 642 A2C Assignment Code

To install: \
Setup virtual environment if needed \
Next:
```
pip install -r requirements.txt
```

To run:
```
python hierarchical_learning.py -a 5 -e 6 -s 2  
```

- a is the number of arenas / trials to execute in parallel
- e is the number of episodes to train
- s is the number of episodes to run before saving to results and weights folders