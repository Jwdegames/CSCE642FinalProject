# Example from http://docs.gym.derkgame.com/#installing

from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path

env = DerkEnv(n_arenas=16, reward_function={"damageEnemyUnit": 1, "damageEnemyStatue": 2, "killEnemyStatue": 4, 
                "killEnemyUnit": 1, "healFriendlyStatue": 1, "healTeammate1": 1, "healTeammate2": 1,
                "timeSpentHomeBase": -1, "timeSpentHomeTerritory": -1, "timeSpentAwayTerritory": 1, "timeSpentAwayBase": 1,
                "damageTaken": -1, "friendlyFire": -2, "healEnemy": -2, "fallDamageTaken": -1, "statueDamageTaken": -1, "timeScaling": 0.9}, turbo_mode=True)

class Network:
    def __init__(self, weights=None, biases=None, mode="MACRO"):
        self.mode = mode 
        if mode == "MACRO":
            # Macro actions are focuses
            self.network_outputs = 6
        else:
            # Micro actions are movex, castx, etc
            self.network_outputs = 7
        if weights is None:
            if mode == "MACRO":
                weights_shape = (self.network_outputs, len(ObservationKeys))
            else:
                # Micro actions will take the focus as an input
                weights_shape = (self.network_outputs, len(ObservationKeys) + 1)
            self.weights = np.random.normal(size=weights_shape)
        else:
            self.weights = weights
        if biases is None:
            self.biases = np.random.normal(size=(self.network_outputs))
        else:
            self.biases = biases

    def clone(self):
        return Network(np.copy(self.weights), np.copy(self.biases))

    def forward(self, observations):
        outputs = np.add(np.matmul(self.weights, observations), self.biases)
        if self.mode == "MACRO":
            # Only focus is the output
            focuses = outputs 
            focus_i = np.argmax(focuses)
            return (focus_i + 1) if focuses[focus_i] > 0 else 0, # Focus
        else:
            # Handles MoveX, Rotate, ChaseFocus, and CastSlot
            casts = outputs[3:6]
            cast_i = np.argmax(casts)
            return (
                math.tanh(outputs[0]), # MoveX
                math.tanh(outputs[1]), # Rotate
                max(min(outputs[2], 1), 0), # ChaseFocus
                (cast_i + 1) if casts[cast_i] > 0 else 0, # CastSlot
            )

    def copy_and_mutate(self, network, mr=0.1):
        self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
        self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)
        
class HierarchicalNetwork:
    def __init__(self, macro_weights=None, micro_weights=None, macro_biases=None, micro_biases=None, cloning=False, macro_network=None, micro_network=None):
        if not cloning:
            self.macro_network = Network(macro_weights, macro_biases, "MACRO")
            self.micro_network = Network(micro_weights, micro_biases, "MICRO")
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

    

macro_weights = np.load('weights/macro_weights.npy') if os.path.isfile('weights/macro_weights.npy') else None
micro_weights = np.load('weights/micro_weights.npy') if os.path.isfile('weights/micro_weights.npy') else None
macro_biases = np.load('weights/macro_biases.npy') if os.path.isfile('weights/macro_biases.npy') else None
micro_biases = np.load('weights/micro_biases.npy') if os.path.isfile('weights/micro_biases.npy') else None
networks = [HierarchicalNetwork(macro_weights, micro_weights, macro_biases, micro_biases) for i in range(env.n_agents)] 


previous_top_network = None
previous_top_reward = None
for e in range(10):
    env.mode = "train"
    print(env.mode)
    print(env.episode_stats)
    observation_n = env.reset()
    while True:
        # Select macro action
        macro_actions = [networks[i].macro_forward(observation_n[i]) for i in range(env.n_agents)]
        # Select micro action
        micro_actions = [networks[i].micro_forward(np.append(observation_n[i], macro_actions[i])) for i in range(env.n_agents)]
        # Set the action to send to the environment
        actions = [micro_actions[i] + macro_actions[i] for i in range(env.n_agents)]
        # Run the action
        observation_n, reward_n, done_n, info = env.step(actions)
        terminal = all(done_n)
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
env.close()