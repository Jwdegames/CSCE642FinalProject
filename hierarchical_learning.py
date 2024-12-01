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
                "damageTaken": -1, "friendlyFire": -2, "healEnemy": -2, "fallDamageTaken": -1, "statueDamageTaken": -1, "timeScaling": 0.9}, turbo_mode=True)

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
            self.network_outputs = 10
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
        observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
        inputs = torch.cat([observation_tensor], dim = -1)
        inputs = inputs.to(cuda_device)
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
            # CastSlot is discrete, get action probabilities
            probabilities = F.softmax(self.layers[-2](outputs)[6:], dim = -1)
            # Everything else is continuous, get mean and standard deviation
            sigmoid_indices = torch.LongTensor([0, 2, 4])
            continuous_mean_values = F.sigmoid(self.layers[-2](outputs)[sigmoid_indices])
            relu_indices = torch.LongTensor([1, 3, 5])
            continuous_std_values = torch.add(F.relu(self.layers[-2](outputs)[relu_indices]), 1e-10)
            continuous_values = torch.cat((continuous_mean_values, continuous_std_values))
            value = self.layers[-1](outputs)
            return probabilities, continuous_values, value

    def copy_and_mutate(self, network, mr=0.1):
        self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
        self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)
        
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
        probabilities, continuous_values, value = self.hierarchical_network.micro_network(observations)
        probabilities_np = probabilities.cpu().detach().numpy()
        cast_action = np.random.choice(len(probabilities_np), p=probabilities_np)
        cast_probability = probabilities[cast_action]
        continuous_values_np = continuous_values.cpu().detach().numpy()
        # Sigmoid was applied to moveX_mean, rotate_mean -> subtract 0.5 and multiply by 2 to get range of [-1, 1]
        moveX_mean = (continuous_values_np[0] - 0.5) * 2
        moveX_std = continuous_values_np[3] 
        rotate_mean = (continuous_values_np[1] - 0.5) * 2
        rotate_std = continuous_values_np[4] 
        chase_mean = continuous_values_np[2]
        chase_std = continuous_values_np[5] 
        # Generate normal values
        values = scipy.stats.norm.rvs(loc=[moveX_mean, rotate_mean, chase_mean], scale=[moveX_std, rotate_std, chase_std])
        # Calculate probabilities for the normal values
        probabilities = scipy.stats.norm.pdf(values, [moveX_mean, rotate_mean, chase_mean], [moveX_std, rotate_std, chase_std])
        joint_probability = cast_probability * probabilities[0] * probabilities[1] * probabilities[2]
        return (np.clip(values[0], -1, 1), np.clip(values[1], -1, 1), np.clip(values[2], 0, 1), cast_action), joint_probability, value
        # return (0, 0, 1, cast_action), cast_probability, value
    
    def actor_loss(self, advantage, probabilities):
        """Calculates actor loss"""
        loss = advantage * -torch.log(probabilities)
        return loss 
    
    def critic_loss(self, advantage, value):
        """Calculates critic loss"""
        loss = advantage * -value
        return loss
    
    def update_macro_actor_critic(self, advantage, probabilities, value):
        """Updates optimizer for macro actor critic"""
        # Compute loss
        actor_loss = self.actor_loss(advantage.detach(), probabilities).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()
        loss = actor_loss + critic_loss

        # Update macro actor critic
        self.macro_optimizer.zero_grad()
        loss.backward()
        self.macro_optimizer.step()
        
    def update_micro_actor_critic(self, advantage, probabilities, value):
        """Updates optimizer for macro actor critic"""
        # Compute loss
        actor_loss = self.actor_loss(advantage.detach(), probabilities).mean()
        critic_loss = self.critic_loss(advantage.detach(), value).mean()
        loss = actor_loss + critic_loss

        # Update micro actor critic
        self.micro_optimizer.zero_grad()
        loss.backward()
        self.micro_optimizer.step()
        

        

    
hidden_layer_sizes = [16, 16]
hidden_layer_string = f"{hidden_layer_sizes[0]}"
for i in range(1, len(hidden_layer_sizes)):
    hidden_layer_string += f"_{hidden_layer_sizes[i]}"
macro_model_exists = f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt' if os.path.isfile(f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt') else None
micro_model_exists = f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt' if os.path.isfile(f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt') else None
if macro_model_exists == None:
    macro_networks = [Network(hidden_layer_sizes, mode="MACRO") for i in range(env.n_agents)] 
else:
    macro_networks = [torch.load(macro_model_exists, weights_only=False) for i in range(env.n_agents)]
if micro_model_exists == None:
    micro_networks = [Network(hidden_layer_sizes, mode="MICRO") for i in range(env.n_agents)] 
else:
    micro_networks = [torch.load(micro_model_exists, weights_only=False) for i in range(env.n_agents)]
networks = [HierarchicalA2CSolver(hidden_layer_sizes, hidden_layer_sizes, cloning=True, macro_network=macro_networks[i], micro_network=micro_networks[i]) for i in range(env.n_agents)] 


# for network in networks:
#     network.hierarchical_network.macro_network = network.hierarchical_network.macro_network.to(cuda_device)
#     network.hierarchical_network.micro_network = network.hierarchical_network.micro_network.to(cuda_device)
gamma = 0.9
previous_top_network = None
previous_top_reward = None
reward_n = None

for e in range(100):
    # env.mode = "train"
    print(env.mode)
    observation_n = env.reset()
    num_networks = len(networks)
    probability_value_pairs = None
    while True:
        appended_actions = []
        # Select macro action
        macro_actions = [networks[i].select_macro_action(observation_n[i]) for i in range(num_networks)]
        # Select micro action
        micro_actions = [networks[i].select_micro_action(np.append(observation_n[i], macro_actions[i][0])) for i in range(num_networks)]
        # Set the action to send to the environment
        # print(micro_actions[0][0])
        # print(macro_actions[0][0])
        actions = [micro_actions[i][0] + (macro_actions[i][0], ) for i in range(num_networks)]
        # Run the action
        observation_n, reward_n, done_n, info = env.step(actions)
        # Compute the TD Error
        for i in range(num_networks):
            probability_value_pair = (networks[i].hierarchical_network.macro_network(observation_n[i]),
                                        networks[i].hierarchical_network.micro_network(np.append(observation_n[i], macro_actions[i][0])))
            td_errors = (reward_n[i] - macro_actions[i][2], reward_n[i] - micro_actions[i][2])
            
            terminal = all(done_n)
            if not terminal:
                # Update td error if not terminal
                td_errors = (td_errors[0] + gamma * probability_value_pair[0][1], td_errors[1] + gamma * probability_value_pair[1][2])
            # Update actor critics
            networks[i].update_macro_actor_critic(td_errors[0], macro_actions[i][1], macro_actions[i][2])
            networks[i].update_micro_actor_critic(td_errors[1], micro_actions[i][1], micro_actions[i][2])
        if terminal:
            print(f"Episode {e} finished")
            break

    print(env.episode_stats)      
    if env.mode == 'train':
        # Update other networks based on best one
        reward_n = env.total_reward
        top_network_i = np.argmax(reward_n)
        top_network = networks[top_network_i].clone()
        print('Current Episode Top Reward', reward_n[top_network_i])
        # Determine if this network or the previous top one is the best network
        if previous_top_reward != None:
            if previous_top_reward > reward_n[top_network_i]:
                top_network = previous_top_network
                print("Top reward is still", previous_top_reward)
            else:
                previous_top_reward = reward_n[top_network_i]
                previous_top_network = top_network
                print("This is the new top reward")
        else:
            previous_top_reward = reward_n[top_network_i]
            previous_top_network = top_network
            print("This is the first top reward")
            
        # Copy and mutate the networks to be based off othe top one
        for network in networks:
            network.copy_and_mutate(top_network)

        # Save the weights of the top network
        np.save('weights/macro_weights.npy', top_network.macro_network.weights)
        np.save('weights/macro_biases.npy', top_network.macro_network.biases)
        np.save('weights/micro_weights.npy', top_network.micro_network.weights)
        np.save('weights/micro_biases.npy', top_network.micro_network.biases)
        
# Save the best network
reward_n = env.total_reward
top_network_i = np.argmax(reward_n)
top_network = networks[top_network_i]
print("Top reward was", reward_n[top_network_i], "for network", top_network_i)
torch.save(top_network.hierarchical_network.macro_network, f'weights/macro_hierarchical_a2c_{hidden_layer_string}.pt')
torch.save(top_network.hierarchical_network.micro_network, f'weights/micro_hierarchical_a2c_{hidden_layer_string}.pt')
env.close()