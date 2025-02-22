import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Initialize a target network and initialize the target network to the policy net
        ### CODE ###
        self.target_net = DQN(action_size).to(device)
        self.target_net.eval()
        self.update_target_net()
        
        
    def load_policy_net(self, path):
        self.policy_net = torch.load(path)           

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        ### CODE ###
        policy_dict = self.policy_net.state_dict()
        self.target_net.load_state_dict(policy_dict)


    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE #### (copy over from agent.py!)
            a = random.randrange(self.action_size)
        else:
            ### CODE #### (copy over from agent.py!)
            state = torch.from_numpy(state).float().unsqueeze(0)
            state = state.to(device)
            # best action
            Q = self.policy_net(state)
            a = torch.argmax(Q).item()
            
        return torch.tensor(a)

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)
        
        # Your agent.py code here with double DQN modifications
         
        ### CODE ####
        current_action = self.policy_net(states)
        idx = actions.unsqueeze(1)
        current_val = current_action.gather(1, idx).squeeze(1)
        
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).to(device)
            next_action = self.target_net(next_states)
            # Find maximum Q-value of action at next state from policy net
            ### CODE ####
            max_action = next_action.max(1)
            max_val = max_action[0].detach()
        
        ### CODE ####

        next_val = mask.float().to(device) * max_val # do not require gradients
        expect_val = next_val * self.discount_factor + rewards
        loss = nn.functional.smooth_l1_loss(current_val, expect_val)
        
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
     
        