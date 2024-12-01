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


cuda_device = torch.device("cpu")
# if torch.cuda.is_available():
#     cuda_device = torch.device("cuda")
#     print(f"Will be using cuda device named {torch.cuda.get_device_name(0)}")
# else:
#     print("No cuda device available. Using CPU")

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

class Network(nn.Module):
    def __init__(self, hidden_layer_sizes, mode="MACRO"):
        super().__init__()
        self.mode = mode 
        # Make the network
        self.layers = nn.ModuleList()
        if mode == "MACRO":
            # Macro actions are focuses
            self.network_outputs = 8
            self.observation_inputs = len(ObservationKeys)
        else:
            # Micro actions are Movex, Rotate, ChaseFocus, CastSlot
            # MoveX discrete actions = [-1, 0, 1]
            # Rotate discrete actions = [-1, -0.5, 0, 0.5, 1]
            # ChaseFocus discrete actions = [0, 0.5, 1]
            # CastSlot discrete actions = [0, 1, 2, 3]
            self.network_outputs = 15
            # self.continuous_network_outputs = 6
            self.observation_inputs = len(ObservationKeys) + 1
        self.layer_sizes = [self.observation_inputs] + hidden_layer_sizes + [self.network_outputs]
        for idx in range(len(self.layer_sizes) - 2):
            self.layers.append(nn.Linear(self.layer_sizes[idx], self.layer_sizes[idx + 1]))
        # Handle the actor layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], self.network_outputs))
        # Handle the critic layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], 1))
        

    def clone(self):
        return copy.deepcopy(self)

    def forward(self, observations):
        inputs = torch.cat([observations], dim = -1)
        for idx in range(len(self.layers) - 2):
            outputs = F.relu(self.layers[idx](inputs))
            inputs = outputs
        if self.mode == "MACRO":
            # Only focus is the output
            # Get the discrete action probabilities and value
            probabilities = F.softmax(self.layers[-2](outputs), dim = -1)
            value = self.layers[-1](outputs)
            return probabilities, value
        else:
            # Handles MoveX, Rotate, ChaseFocus, and CastSlot
            layer_outputs = self.layers[-2](outputs)
            probabilities_moveX = F.softmax(layer_outputs[0:3], dim = -1)
            probabilities_rotate = F.softmax(layer_outputs[3:8], dim = -1)
            probabilities_chaseFocus = F.softmax(layer_outputs[8:11], dim = -1)
            probabilities_castSlot = F.softmax(layer_outputs[11:], dim = -1)
            value = self.layers[-1](outputs)
            return (probabilities_moveX, probabilities_rotate, probabilities_chaseFocus, probabilities_castSlot), value
        
class HierarchicalNetwork:
    def __init__(self, macro_hidden_layer_sizes, micro_hidden_layer_sizes, cloning=False, macro_network=None, micro_network=None):
        if not cloning:
            self.macro_network = Network(macro_hidden_layer_sizes, "MACRO")
            self.micro_network = Network(micro_hidden_layer_sizes, "MICRO")
        else:
            self.macro_network = macro_network
            self.micro_network = micro_network
        
    def macro_forward(self, observations):
        return self.macro_network.forward(observations)
    
    def micro_forward(self, observations):
        return self.micro_network.forward(observations)
    
    def clone(self):
        cloned_network = HierarchicalNetwork(cloning=True, macro_network=self.macro_network.clone(), micro_network=self.micro_network.clone())
        return cloned_network
    
    def copy_and_mutate(self, network, mr=0.1):
        self.macro_network.copy_and_mutate(network.macro_network, mr)
        self.micro_network.copy_and_mutate(network.micro_network, mr)

class HierarchicalA2CSolver:
    "Performs A2C on hierarchical network"
    def __init__(self, macro_hidden_layer_sizes, micro_hidden_layer_sizes, cloning=False, macro_network=None, micro_network=None, learning_rate = 0.001):
        self.hierarchical_network = HierarchicalNetwork(macro_hidden_layer_sizes, micro_hidden_layer_sizes, cloning, macro_network, micro_network)
        self.macro_optimizer = Adam(self.hierarchical_network.macro_network.parameters(), lr=learning_rate)
        self.micro_optimizer = Adam(self.hierarchical_network.micro_network.parameters(), lr=learning_rate)
    
    def select_macro_action(self, observations):
        """Selects a macro action"""
        probabilities, value = self.hierarchical_network.macro_network(observations)
        probabilities_np = probabilities.cpu().detach().numpy()
        action = np.random.choice(len(probabilities_np), p=probabilities_np)
        return action, probabilities[action], value
    
    def select_micro_action(self, observations):
        """Selects a micro action"""
        probabilities, value = self.hierarchical_network.micro_network(observations)
        moveX_probabilities_np = probabilities[0].cpu().detach().numpy()
        moveX_action = np.random.choice(len(moveX_probabilities_np), p=moveX_probabilities_np)
        moveX_probability = probabilities[0][moveX_action]
        rotate_probabilities_np = probabilities[1].cpu().detach().numpy()
        rotate_action = np.random.choice(len(rotate_probabilities_np), p=rotate_probabilities_np)
        rotate_probability = probabilities[1][rotate_action]
        chaseFocus_probabilities_np = probabilities[2].cpu().detach().numpy()
        chaseFocus_action = np.random.choice(len(chaseFocus_probabilities_np), p=chaseFocus_probabilities_np)
        chaseFocus_probability = probabilities[2][chaseFocus_action]
        cast_probabilities_np = probabilities[3].cpu().detach().numpy()
        cast_action = np.random.choice(len(cast_probabilities_np), p=cast_probabilities_np)
        cast_probability = probabilities[3][cast_action]
        moveX_dictionary = {
            0: -1,
            1: 0,
            2: 1,
            3: 0.5,
            4: 1
        }
        rotate_dictionary = {
            0: -1,
            1: -0.5,
            2: 0,
            3: 0.5,
            4: 1
        }
        chaseFocus_dictionary = {
            0: 0,
            1: 0.5,
            2: 1
        }
        joint_probability = moveX_probability * rotate_probability * chaseFocus_probability * cast_probability
        return (moveX_dictionary[moveX_action], rotate_dictionary[rotate_action], chaseFocus_dictionary[chaseFocus_action], cast_action), joint_probability, value
    
    def get_action(self, observations):
        """Gets an action by getting both macro and micro action"""
        macro_action =  self.select_macro_action(observations)
        micro_action = self.select_micro_action(torch.cat((observations, torch.as_tensor([macro_actions[0]]).to(cuda_device))))
        action = micro_action[0] + (macro_action[0],)
        self.macro_probability = macro_action[1]
        self.macro_value = macro_action[2]
        self.micro_probability = micro_action[1]
        self.micro_value = micro_action[2]
        return action
        
    def actor_loss(self, advantage, probabilities):
        """Calculates actor loss"""
        loss = advantage * -torch.log(probabilities)
        return loss 
    
    def critic_loss(self, advantage, value):
        """Calculates critic loss"""
        loss = advantage * -value
        return loss
    
    def update_macro_actor_critic(self, advantages, probabilities, values):
        """Updates optimizer for macro actor critic"""
        # Compute loss
        advantage = advantages[-1]
        probability = probabilities[-1]
        value = values[-1]
            
        # print(probability, value)
        # print(advantage)
        # print(advantages)
        actor_loss = self.actor_loss(advantage.detach(), probability).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()
        loss = actor_loss + critic_loss
        # print(actor_loss)
        # print(critic_loss)
        # print(loss)

        # Update macro actor critic
        self.macro_optimizer.zero_grad()
        loss.backward()
        self.macro_optimizer.step()
        
    def update_micro_actor_critic(self, advantages, probabilities, values):
        """Updates optimizer for macro actor critic"""
        # Compute loss
        advantage = advantages[-1]
        probability = probabilities[-1]
        value = values[-1]
            
        # print("Advantage:", advantage)
        actor_loss = self.actor_loss(advantage.detach(), probability).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()
        loss = actor_loss + critic_loss

        # Update micro actor critic
        self.micro_optimizer.zero_grad()
        loss.backward()
        self.micro_optimizer.step()
        

        

batch_size = 16
hidden_layer_sizes = [32, 16, 16]
hidden_layer_string = f"{hidden_layer_sizes[0]}"
for i in range(1, len(hidden_layer_sizes)):
    hidden_layer_string += f"_{hidden_layer_sizes[i]}"
macro_model_exists = f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt' if os.path.isfile(f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt') else None
micro_model_exists = f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt' if os.path.isfile(f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt') else None
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
networks = [HierarchicalA2CSolver(hidden_layer_sizes, hidden_layer_sizes, cloning=True, macro_network=macro_networks[i], micro_network=micro_networks[i]) for i in range(env.n_agents)] 


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
        # Select macro action
        macro_actions = [networks[i].select_macro_action(observation_tensor[i]) for i in range(num_networks)]
        # Select micro action
        micro_actions = [networks[i].select_micro_action(torch.cat((observation_tensor[i], torch.as_tensor([macro_actions[i][0]]).to(cuda_device)))) for i in range(num_networks)]
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
                                                
                    td_errors = (reward_n[i] - macro_actions[i][2], reward_n[i] - micro_actions[i][2])
                    
                    batch_terminal = all(done_n)
                    if not batch_terminal:
                        # Update td error if not terminal
                        td_errors = (td_errors[0] + gamma * macro_probability_value_pair[2], td_errors[1] + gamma * micro_probability_value_pair[1])
                    
                    macro_td_batch_errors[i] = td_errors[0]
                    micro_td_batch_errors[i] = td_errors[1]
                    batch_macro_probabilities[i] = macro_actions[i][1]  
                    batch_macro_values[i] = macro_actions[i][2]
                    batch_micro_probabilities[i] = micro_actions[i][1]
                    batch_micro_values[i] = micro_actions[i][2]
                    
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