# Example from http://docs.gym.derkgame.com/#installing


from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path
import scipy.stats
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from hierarchical_classes import Network, HierarchicalA2CSolver, HierarchicalNetwork

cuda_device = torch.device("cpu")
env = DerkEnv(n_arenas=4, reward_function={"damageEnemyUnit": 1, "damageEnemyStatue": 2, "killEnemyStatue": 4, 
                "killEnemyUnit": 1, "healFriendlyStatue": 1, "healTeammate1": 1, "healTeammate2": 1,
                "timeSpentHomeBase": -1, "timeSpentHomeTerritory": -1, "timeSpentAwayTerritory": 1, "timeSpentAwayBase": 1,
                "damageTaken": -1, "friendlyFire": -2, "healEnemy": -2, "fallDamageTaken": -1, "statueDamageTaken": -1, "timeScaling": 0.9},
                home_team=[
                    { 'primaryColor': '#ff0000', 'slots': ['Talons', 'IronBubblegum', 'HeliumBubblegum'], },
                    { 'primaryColor': '#00ffff', 'slots': ['Pistol', 'ParalyzingDart', 'HealingGland'] },
                    { 'primaryColor': '#00ff00', 'slots': ['BloodClaws', 'Cripplers', 'FrogLegs']  },
                ],
                away_team=[
                    { 'primaryColor': '#ff0000', 'slots': ['Talons', 'IronBubblegum', 'HeliumBubblegum'], },
                    { 'primaryColor': '#00ffff', 'slots': ['Pistol', 'ParalyzingDart', 'HealingGland'] },
                    { 'primaryColor': '#00ff00', 'slots': ['BloodClaws', 'Cripplers', 'FrogLegs']  },
                ],
                turbo_mode=True)

    
    

        

        

batch_size = 16
hidden_layer_sizes = [32, 16, 16]
hidden_layer_string = f"{hidden_layer_sizes[0]}"
for i in range(1, len(hidden_layer_sizes)):
    hidden_layer_string += f"_{hidden_layer_sizes[i]}"
# macro_model_exists = f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt' if os.path.isfile(f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt') else None
# micro_model_exists = f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt' if os.path.isfile(f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt') else None
macro_model_exists = None
micro_model_exists = None
if macro_model_exists == None:
    macro_networks = [Network(hidden_layer_sizes, mode="MACRO") for i in range(env.n_agents)] 
else:
    macro_networks = [torch.load(macro_model_exists, weights_only=False) for i in range(env.n_agents)]
if micro_model_exists == None:
    micro_networks = [Network(hidden_layer_sizes, mode="MICRO") for i in range(env.n_agents)] 
else:
    micro_networks = [torch.load(micro_model_exists, weights_only=False) for i in range(env.n_agents)]
networks = [HierarchicalA2CSolver(hidden_layer_sizes, hidden_layer_sizes, cloning=False, macro_network=macro_networks[i], micro_network=micro_networks[i]) for i in range(env.n_agents)] 


for network in networks:
    network.hierarchical_network.macro_network = network.hierarchical_network.macro_network.to(cuda_device)
    network.hierarchical_network.micro_network = network.hierarchical_network.micro_network.to(cuda_device)
gamma = 0.9
previous_top_network = None
previous_top_reward = None
reward_n = None

for e in range(100):
    # env.mode = "train"
    observation_n = env.reset()
    observation_tensor = torch.as_tensor(observation_n, dtype=torch.float32)
    observation_tensor = observation_tensor.to(cuda_device)
    num_networks = len(networks)
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
        actions = [networks[i].get_action(observation_tensor[i]) for i in range(num_networks)]
        # Run the action
        observation_n, reward_n, done_n, info = env.step(actions)
        observation_tensor = torch.as_tensor(observation_n, dtype=torch.float32)
        observation_tensor = observation_tensor.to(cuda_device)
        batch_idx += 1
        if batch_idx == batch_size:
            for i in range(num_networks):
                if isinstance(networks[i], HierarchicalA2CSolver):
                    macro_probability_value_pair = networks[i].select_macro_action(observation_tensor[i])
                    micro_probability_value_pair = networks[i].hierarchical_network.micro_network(torch.cat((observation_tensor[i], torch.as_tensor([macro_probability_value_pair[0]]).to(cuda_device))))
                                                
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
        
# Save the best network
reward_n = env.total_reward
top_network_i = np.argmax(reward_n)
top_network = networks[top_network_i]
print("Top reward was", reward_n[top_network_i], "for network", top_network_i)
torch.save(top_network.hierarchical_network.macro_network, f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt')
torch.save(top_network.hierarchical_network.micro_network, f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt')
env.close()