from gym_derk import ObservationKeys
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Network(nn.Module):
    def __init__(self, hidden_layer_sizes, mode="MACRO"):
        super().__init__()
        self.mode = mode 
        # Make the network
        self.layers = nn.ModuleList()
        # print(f"Making {mode} network")
        if mode == "MACRO":
            # Macro actions are focuses
            self.network_outputs = 8
            self.observation_inputs = len(ObservationKeys)
        elif mode == "MICRO":
            # Micro actions are Movex, Rotate, ChaseFocus, CastSlot
            # MoveX discrete actions = [-1, 0, 1]
            # Rotate discrete actions = [-1, -0.5, 0, 0.5, 1]
            # ChaseFocus discrete actions = [0, 0.5, 1]
            # CastSlot discrete actions = [0, 1, 2, 3]
            self.network_outputs = 15
            self.observation_inputs = len(ObservationKeys) + 1
        elif mode == "SARL_MACRO":
            # Macro actions are focuses
            self.network_outputs = 8 * 3
            self.observation_inputs = len(ObservationKeys) * 3
        elif mode == "SARL_MICRO":
            self.network_outputs = 15 * 3
            self.observation_inputs = len(ObservationKeys) * 3 + 3
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
        elif self.mode == "MICRO":
            # Handles MoveX, Rotate, ChaseFocus, and CastSlot
            layer_outputs = self.layers[-2](outputs)
            probabilities_moveX = F.softmax(layer_outputs[0:3], dim = -1)
            probabilities_rotate = F.softmax(layer_outputs[3:8], dim = -1)
            probabilities_chaseFocus = F.softmax(layer_outputs[8:11], dim = -1)
            probabilities_castSlot = F.softmax(layer_outputs[11:], dim = -1)
            value = self.layers[-1](outputs)
            return (probabilities_moveX, probabilities_rotate, probabilities_chaseFocus, probabilities_castSlot), value
        elif self.mode == "SARL_MACRO":
            # Three different agents
            # Only focus is the output
            # Get the discrete action probabilities and value
            layer_outputs = self.layers[-2](outputs)
            probabilities_agent1 = F.softmax(layer_outputs[0:7], dim = -1)
            probabilities_agent2 = F.softmax(layer_outputs[7:15], dim = -1)
            probabilities_agent3 = F.softmax(layer_outputs[15:], dim = -1)
            value = self.layers[-1](outputs)
            return (probabilities_agent1, probabilities_agent2, probabilities_agent3), value
        elif self.mode == "SARL_MICRO":
            # Three different agents
            # Handles MoveX, Rotate, ChaseFocus, and CastSlot
            layer_outputs = self.layers[-2](outputs)
            probabilities_moveX1 = F.softmax(layer_outputs[0:3], dim = -1)
            probabilities_rotate1 = F.softmax(layer_outputs[3:8], dim = -1)
            probabilities_chaseFocus1 = F.softmax(layer_outputs[8:11], dim = -1)
            probabilities_castSlot1 = F.softmax(layer_outputs[11:15], dim = -1)
            probabilities_moveX2 = F.softmax(layer_outputs[15:18], dim = -1)
            probabilities_rotate2 = F.softmax(layer_outputs[18:23], dim = -1)
            probabilities_chaseFocus2 = F.softmax(layer_outputs[23:26], dim = -1)
            probabilities_castSlot2 = F.softmax(layer_outputs[26:30], dim = -1)
            probabilities_moveX3 = F.softmax(layer_outputs[30:33], dim = -1)
            probabilities_rotate3 = F.softmax(layer_outputs[33:38], dim = -1)
            probabilities_chaseFocus3 = F.softmax(layer_outputs[38:41], dim = -1)
            probabilities_castSlot3 = F.softmax(layer_outputs[41:], dim = -1)
            value = self.layers[-1](outputs)
            return (probabilities_moveX1, probabilities_rotate1, probabilities_chaseFocus1, probabilities_castSlot1,
                    probabilities_moveX2, probabilities_rotate2, probabilities_chaseFocus2, probabilities_castSlot2,
                    probabilities_moveX3, probabilities_rotate3, probabilities_chaseFocus3, probabilities_castSlot3,), value
        
class HierarchicalNetwork:
    def __init__(self, macro_hidden_layer_sizes, micro_hidden_layer_sizes, cloning=False, macro_network=None, micro_network=None, mode ="MARL"):
        self.mode = mode
        if not cloning:
            if mode == "MARL":
                self.macro_network = Network(macro_hidden_layer_sizes, "MACRO")
                self.micro_network = Network(micro_hidden_layer_sizes, "MICRO")
            elif mode == "SARL":
                self.macro_network = Network(macro_hidden_layer_sizes, "SARL_MACRO")
                self.micro_network = Network(micro_hidden_layer_sizes, "SARL_MICRO")
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
    

class HierarchicalA2CSolver:
    "Performs A2C on hierarchical network"
    def __init__(self, macro_hidden_layer_sizes, micro_hidden_layer_sizes, cloning=False, macro_network=None, micro_network=None, learning_rate = 0.001, mode="MARL"):
        self.mode = mode
        self.hierarchical_network = HierarchicalNetwork(macro_hidden_layer_sizes, micro_hidden_layer_sizes, cloning, macro_network, micro_network, mode)
        self.macro_optimizer = Adam(self.hierarchical_network.macro_network.parameters(), lr=learning_rate)
        self.micro_optimizer = Adam(self.hierarchical_network.micro_network.parameters(), lr=learning_rate)
    
    def select_macro_action(self, observations):
        """Selects a macro action"""
        if self.mode == "MARL":
            probabilities, value = self.hierarchical_network.macro_network(observations)
            probabilities_np = probabilities.cpu().detach().numpy()
            action = np.random.choice(len(probabilities_np), p=probabilities_np)
            return action, probabilities[action], value
        elif self.mode == "SARL":
            probabilities, value = self.hierarchical_network.macro_network(observations)
            probabilities0_np = probabilities[0].cpu().detach().numpy()
            action0 = np.random.choice(len(probabilities0_np), p=probabilities0_np)
            probabilities1_np = probabilities[1].cpu().detach().numpy()
            action1 = np.random.choice(len(probabilities1_np), p=probabilities1_np)
            probabilities2_np = probabilities[2].cpu().detach().numpy()
            action2 = np.random.choice(len(probabilities2_np), p=probabilities2_np)
            joint_probability = probabilities[0][action0] * probabilities[1][action1] * probabilities[2][action2]
            return (action0, action1, action2), joint_probability, value
    
    def select_micro_action(self, observations):
        """Selects a micro action"""
        if self.mode == "MARL":
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
        elif self.mode == "SARL":
            # Agent 0
            probabilities, value = self.hierarchical_network.micro_network(observations)
            moveX_probabilities_np0 = probabilities[0].cpu().detach().numpy()
            moveX_action0 = np.random.choice(len(moveX_probabilities_np0), p=moveX_probabilities_np0)
            moveX_probability0 = probabilities[0][moveX_action0]
            rotate_probabilities_np0 = probabilities[1].cpu().detach().numpy()
            rotate_action0 = np.random.choice(len(rotate_probabilities_np0), p=rotate_probabilities_np0)
            rotate_probability0 = probabilities[1][rotate_action0]
            chaseFocus_probabilities_np0 = probabilities[2].cpu().detach().numpy()
            chaseFocus_action0 = np.random.choice(len(chaseFocus_probabilities_np0), p=chaseFocus_probabilities_np0)
            chaseFocus_probability0 = probabilities[2][chaseFocus_action0]
            cast_probabilities_np0 = probabilities[3].cpu().detach().numpy()
            cast_action0 = np.random.choice(len(cast_probabilities_np0), p=cast_probabilities_np0)
            cast_probability0 = probabilities[3][cast_action0]
            
            
            # Agent 1
            moveX_probabilities_np1 = probabilities[4].cpu().detach().numpy()
            moveX_action1 = np.random.choice(len(moveX_probabilities_np1), p=moveX_probabilities_np1)
            moveX_probability1 = probabilities[0][moveX_action1]
            rotate_probabilities_np1 = probabilities[5].cpu().detach().numpy()
            rotate_action1 = np.random.choice(len(rotate_probabilities_np1), p=rotate_probabilities_np1)
            rotate_probability1 = probabilities[1][rotate_action1]
            chaseFocus_probabilities_np1 = probabilities[6].cpu().detach().numpy()
            chaseFocus_action1 = np.random.choice(len(chaseFocus_probabilities_np1), p=chaseFocus_probabilities_np1)
            chaseFocus_probability1 = probabilities[2][chaseFocus_action1]
            cast_probabilities_np1 = probabilities[7].cpu().detach().numpy()
            cast_action1 = np.random.choice(len(cast_probabilities_np1), p=cast_probabilities_np1)
            cast_probability1 = probabilities[3][cast_action1]
            
            # Agent 2
            moveX_probabilities_np2 = probabilities[8].cpu().detach().numpy()
            moveX_action2 = np.random.choice(len(moveX_probabilities_np2), p=moveX_probabilities_np2)
            moveX_probability2 = probabilities[0][moveX_action2]
            rotate_probabilities_np2 = probabilities[9].cpu().detach().numpy()
            rotate_action2 = np.random.choice(len(rotate_probabilities_np2), p=rotate_probabilities_np2)
            rotate_probability2 = probabilities[1][rotate_action2]
            chaseFocus_probabilities_np2 = probabilities[10].cpu().detach().numpy()
            chaseFocus_action2 = np.random.choice(len(chaseFocus_probabilities_np2), p=chaseFocus_probabilities_np2)
            chaseFocus_probability2 = probabilities[2][chaseFocus_action2]
            cast_probabilities_np2 = probabilities[11].cpu().detach().numpy()
            cast_action2 = np.random.choice(len(cast_probabilities_np2), p=cast_probabilities_np2)
            cast_probability2 = probabilities[3][cast_action0]
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
            joint_probability = (moveX_probability0 * rotate_probability0 * chaseFocus_probability0 * cast_probability0 * 
                                 moveX_probability1 * rotate_probability1 * chaseFocus_probability1 * cast_probability1 * 
                                 moveX_probability2 * rotate_probability2 * chaseFocus_probability2 * cast_probability2)
            return ((moveX_dictionary[moveX_action0], rotate_dictionary[rotate_action0], chaseFocus_dictionary[chaseFocus_action0], cast_action0),
                    (moveX_dictionary[moveX_action1], rotate_dictionary[rotate_action1], chaseFocus_dictionary[chaseFocus_action1], cast_action1),
                    (moveX_dictionary[moveX_action2], rotate_dictionary[rotate_action2], chaseFocus_dictionary[chaseFocus_action2], cast_action2)), joint_probability, value
            
    
    def get_action(self, observations):
        """Gets an action by getting both macro and micro action"""
        if self.mode == "MARL":
            macro_action =  self.select_macro_action(observations)
            micro_action = self.select_micro_action(torch.cat((observations, torch.as_tensor([macro_action[0]]))))
            action = micro_action[0] + (macro_action[0],)
            self.macro_probability = macro_action[1]
            self.macro_value = macro_action[2]
            self.micro_probability = micro_action[1]
            self.micro_value = micro_action[2]
            return action
        elif self.mode == "SARL":
            macro_action =  self.select_macro_action(observations)
            micro_action = self.select_micro_action(torch.cat((observations, torch.as_tensor([macro_action[0][0], macro_action[0][1], macro_action[0][2]]))))
            actions = (micro_action[0][0] + (macro_action[0][0],), micro_action[0][1] + (macro_action[0][1],), micro_action[0][2] + (macro_action[0][2],))
            self.macro_probability = macro_action[1]
            self.macro_value = macro_action[2]
            self.micro_probability = micro_action[1]
            self.micro_value = micro_action[2]
            return actions
        
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