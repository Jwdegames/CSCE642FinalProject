# Gym Derk imports are from Derk's Gym. See more at http://docs.gym.derkgame.com/. 
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os
import os.path
import pandas as pd
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from hierarchical_classes import Network, HierarchicalA2CSolver, HierarchicalNetwork
import argparse


parser = argparse.ArgumentParser(
                    prog='MAHRLvsSAHRLDerksGym',
                    description='This program trains Multi-agent hierarchical reinforcement learning architectures against single-agent hierarchical reinforcement learning architectures',
                    epilog='Make sure to follow the readme for instructions. Example execution: python hierarchical_learning.py -a 5 -e 6 -s 2')
parser.add_argument("-a", "--arenas", help='The number of arenas / trials to run in parallel.', required = True)
parser.add_argument("-e", "--episodes", help='The number of episodes to run.', required = True)
parser.add_argument("-s", "--save", help='The number of episodes to save the results and weights after.', required = True)

args = parser.parse_args()
num_arenas = int(args.arenas)
num_episodes = int(args.episodes)
episode_save_batch = int(args.save)
print("Number of arenas:", num_arenas)
print("Number of episodes:", num_episodes)
print("Number of episodes before each save:", episode_save_batch)
start_episode = 0
env = DerkEnv(n_arenas=num_arenas, reward_function={"damageEnemyUnit": 20, "damageEnemyStatue": 20, "killEnemyStatue": 40, 
                "killEnemyUnit": 40, "healFriendlyStatue": 20, "healTeammate1": 10, "healTeammate2": 10,
                "timeSpentHomeBase": -1, "timeSpentHomeTerritory": -1, "timeSpentAwayTerritory": 1, "timeSpentAwayBase": 1,
                "damageTaken": -1, "friendlyFire": -10, "healEnemy": -10, "fallDamageTaken": -20, "statueDamageTaken": -1, "timeScaling": 0.9,
                "teamSpirit": 0.2},
                home_team=[
                    { 'primaryColor': '#ff0000', 'slots': ['Cleavers', 'Cripplers', 'VampireGland'], },
                    { 'primaryColor': '#00ffff', 'slots': ['Blaster', 'HeliumBubblegum', 'HealingGland'] },
                    { 'primaryColor': '#00ff00', 'slots': ['Magnum', 'IronBubblegum', 'ParalyzingDart']  },
                ],
                away_team=[
                    { 'primaryColor': '#ff0000', 'slots': ['Cleavers', 'Cripplers', 'VampireGland'], },
                    { 'primaryColor': '#00ffff', 'slots': ['Blaster', 'HeliumBubblegum', 'HealingGland'] },
                    { 'primaryColor': '#00ff00', 'slots': ['Magnum', 'IronBubblegum', 'ParalyzingDart']  },
                ],
                turbo_mode=True)

    
    

        

        

batch_size = 1
hidden_layer_sizes = [32, 16, 16]
hidden_layer_string = f"{hidden_layer_sizes[0]}"
for i in range(1, len(hidden_layer_sizes)):
    hidden_layer_string += f"_{hidden_layer_sizes[i]}"
networks = [0] * (num_arenas * 4)
for i in range(num_arenas):
    for j in range(4):
        # Create 3 MARL networks on one team, and 1 SARL network for the other team
        if j < 3:
            network_mode = "MARL"
        else:
            network_mode = "SARL"
        networks[i * 4 + j] = HierarchicalA2CSolver(hidden_layer_sizes, hidden_layer_sizes, mode=network_mode)
        if start_episode != 0: 
            networks[i * 4 + j].hierarchical_network.macro_network = torch.load(f'weights/{num_episodes}_{hidden_layer_string}/macro_hierarchical_a2c_{start_episode}_{hidden_layer_string}_{i * 4 + j}_{network_mode}.pt', weights_only=False)
            networks[i * 4 + j].hierarchical_network.micro_network = torch.load(f'weights/{num_episodes}_{hidden_layer_string}/micro_hierarchical_a2c_{start_episode}_{hidden_layer_string}_{i * 4 + j}_{network_mode}.pt', weights_only=False)


if not os.path.isdir(f'weights'):
    os.mkdir(f'weights')
if not os.path.isdir(f'results'):
    os.mkdir(f'results')

if not os.path.isdir(f'weights/{num_episodes}_{hidden_layer_string}'):
    os.mkdir(f'weights/{num_episodes}_{hidden_layer_string}')
if not os.path.isdir(f'results/{num_episodes}_{hidden_layer_string}'):
    os.mkdir(f'results/{num_episodes}_{hidden_layer_string}')




gamma = 0.9
previous_top_network = None
previous_top_reward = None
reward_n = None
num_networks = len(networks)
team_rewards = [[0] * num_episodes for i in range(num_arenas * 2)] 
#print(team_rewards)
player_rewards = [[0] * num_episodes for i in range(env.n_agents)] 
if start_episode != 0:
    team_df = pd.read_csv(f"results/{num_episodes}_{hidden_layer_string}/team_data_{start_episode}_{hidden_layer_string}.csv")
    team_dictionary =  team_df.to_dict()
    for i in range(num_arenas * 2):
        for episode in range(start_episode):
            team_rewards[i][episode] = team_dictionary[f"Team {i}"][episode]
    player_df = pd.read_csv(f"results/{num_episodes}_{hidden_layer_string}/player_data_{start_episode}_{hidden_layer_string}.csv")
    player_dictionary = player_df.to_dict()
    for i in range(env.n_agents):
        for episode in range(start_episode):
            player_rewards[i][episode] = player_dictionary[f"Player {i}"][episode]
for e in range(start_episode, num_episodes):
    # env.mode = "train"
    observation_n = env.reset()
    observation_tensor = torch.as_tensor(observation_n, dtype=torch.float32)
    observation_tensor = observation_tensor
    
    probability_value_pairs = None
    macro_td_batch_errors = [0] * num_networks
    # print(macro_td_batch_errors)
    micro_td_batch_errors = [0] * num_networks
    batch_macro_probabilities = [0] * num_networks
    batch_macro_values = [0] * num_networks
    batch_micro_probabilities = [0] * num_networks
    batch_micro_values = [0] * num_networks
    batch_idx = 0

    while True:
        appended_actions = []
        # Set the action to send to the environment
        actions = [0] * env.n_agents
        observation_idx = 0
        for network_idx in range(num_networks):
            network = networks[network_idx]
            if isinstance(network, HierarchicalA2CSolver):
                if network.mode == "MARL":
                    actions[observation_idx] = network.get_action(observation_tensor[observation_idx])
                    observation_idx += 1
                elif network.mode == "SARL":
                    sarl_observations = torch.cat((observation_tensor[observation_idx], observation_tensor[observation_idx + 1], observation_tensor[observation_idx + 2]))
                    sarl_actions = network.get_action(sarl_observations)
                    actions[observation_idx] = sarl_actions[0]
                    actions[observation_idx + 1] = sarl_actions[1]
                    actions[observation_idx + 2] = sarl_actions[2]
                    observation_idx += 3
        # Run the action
        observation_n, reward_n, done_n, info = env.step(actions)
        observation_tensor = torch.as_tensor(observation_n, dtype=torch.float32)
        observation_tensor = observation_tensor
        batch_idx += 1
        if batch_idx == batch_size:
            observation_idx = 0
            for i in range(num_networks):
                if isinstance(networks[i], HierarchicalA2CSolver):
                    if networks[i].mode == "MARL":
                        macro_probability_value_pair = networks[i].select_macro_action(observation_tensor[observation_idx])
                        micro_probability_value_pair = networks[i].hierarchical_network.micro_network(torch.cat((observation_tensor[observation_idx], torch.as_tensor([macro_probability_value_pair[0]]))))
                        observation_idx += 1
                    elif networks[i].mode == "SARL":
                        sarl_observations = torch.cat((observation_tensor[observation_idx], observation_tensor[observation_idx + 1], observation_tensor[observation_idx + 2]))
                        macro_probability_value_pair = networks[i].select_macro_action(sarl_observations)
                        micro_probability_value_pair = networks[i].hierarchical_network.micro_network(torch.cat((sarl_observations, 
                                                            torch.as_tensor([macro_probability_value_pair[0][0], macro_probability_value_pair[0][1], macro_probability_value_pair[0][2]]))))
                        observation_idx += 3                        
                    td_errors = (reward_n[i] - networks[i].macro_value, reward_n[i] - networks[i].micro_value)
                    
                    batch_terminal = all(done_n)
                    if not batch_terminal:
                        # Update td error if not terminal
                        td_errors = (td_errors[0] + gamma * macro_probability_value_pair[2], td_errors[1] + gamma * micro_probability_value_pair[1])
                    
                    macro_td_batch_errors[i] = td_errors[0]
                    micro_td_batch_errors[i] = td_errors[1]
                    batch_macro_probabilities[i] = networks[i].macro_probability  
                    batch_macro_values[i] = networks[i].macro_value
                    batch_micro_probabilities[i] = networks[i].micro_probability
                    batch_micro_values[i] = networks[i].micro_value
                    
                    # Update actor critics
                    networks[i].update_macro_actor_critic([macro_td_batch_errors[i]], [batch_macro_probabilities[i]], [batch_macro_values[i]])
                    networks[i].update_micro_actor_critic([micro_td_batch_errors[i]], [batch_micro_probabilities[i]], [batch_micro_values[i]])
                    # print(macro_td_batch_errors[i])
                
            macro_td_batch_errors = [0] * num_networks
            micro_td_batch_errors = [0] * num_networks
            batch_macro_probabilities = [0] * num_networks
            batch_macro_values = [0] * num_networks
            batch_micro_probabilities = [0] * num_networks
            batch_micro_values = [0] * num_networks
        terminal = all(done_n)
        if terminal:
            print(f"Episode {e} finished")
            break
    print(env.episode_stats)
    for team_reward_idx in range(len(env.team_stats)):
        # print(team_reward_idx)
        team_stat = env.team_stats[team_reward_idx]
        team_rewards[team_reward_idx][e] = team_stat[0]
    for player_reward_idx in range(env.n_agents):
        player_stat = env.total_reward[player_reward_idx]
        player_rewards[player_reward_idx][e] = player_stat
    if (e + 1) % episode_save_batch == 0:
        # Save the dictonaries and models
        team_dictionary = {}
        for team_reward_idx in range(len(env.team_stats)):
            team_dictionary[f"Team {team_reward_idx}"] = team_rewards[team_reward_idx]
        team_df = pd.DataFrame(data=team_dictionary)
        team_df.to_csv(f"results/{num_episodes}_{hidden_layer_string}/team_data_{e+1}_{hidden_layer_string}.csv")
            
        player_dictionary = {}
        for player_reward_idx in range(env.n_agents):
            player_dictionary[f"Player {player_reward_idx}"] = player_rewards[player_reward_idx]
        player_df = pd.DataFrame(data=player_dictionary)
        player_df.to_csv(f"results/{num_episodes}_{hidden_layer_string}/player_data_{e+1}_{hidden_layer_string}.csv")
        
        for i in range(num_networks):
            torch.save(networks[i].hierarchical_network.macro_network, f'weights/{num_episodes}_{hidden_layer_string}/macro_hierarchical_a2c_{e+1}_{hidden_layer_string}_{i}_{networks[i].mode}.pt')
            torch.save(networks[i].hierarchical_network.micro_network, f'weights/{num_episodes}_{hidden_layer_string}/micro_hierarchical_a2c_{e+1}_{hidden_layer_string}_{i}_{networks[i].mode}.pt')
        

env.close()
# Save the networks
for i in range(num_networks):
    torch.save(networks[i].hierarchical_network.macro_network, f'weights/{num_episodes}_{hidden_layer_string}/macro_hierarchical_a2c_{num_episodes}_{hidden_layer_string}_{i}_{networks[i].mode}.pt')
    torch.save(networks[i].hierarchical_network.micro_network, f'weights/{num_episodes}_{hidden_layer_string}/micro_hierarchical_a2c_{num_episodes}_{hidden_layer_string}_{i}_{networks[i].mode}.pt')
