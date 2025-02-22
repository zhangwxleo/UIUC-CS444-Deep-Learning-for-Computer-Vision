import numpy as np
import torch
import torch.optim as optim
from agent import Agent
from memory import ReplayMemoryLSTM
from model import DQN_LSTM
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_LSTM(Agent):
    def __init__(self, action_size):
        super().__init__(action_size)
        # Generate the memory
        self.memory = ReplayMemoryLSTM()

        # Create the policy net
        self.policy_net = DQN_LSTM(action_size)
        self.policy_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state, hidden=None):
        ### CODE ###
        # Similar to that for Agent
        # You should pass the state and hidden through the policy net even when you are randomly selecting an action so you can get the hidden state for the next state
        # We recommend the following outline:
        # 1. Pass the state and hidden through the policy net. You should pass train=False to the forward function of the policy net here becasue you are not training the policy net here
        # 2. If you are randomly selecting an action, return the random action and policy net's hidden, otherwise return the policy net's action and hidden
        if np.random.rand() <= self.epsilon:
            a = np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0)
                state = state.to(device)
                Q, hidden = self.policy_net(state)
                a = torch.argmax(Q).item()
                
        return torch.tensor(a), hidden
    

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :lstm_seq_length, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        mask = torch.tensor(list(map(int, dones==False)),dtype=torch.uint8)

        ### All the following code is nearly same as that for Agent

        # Compute Q(s_t, a), the Q-value of the current state
        # You should hidden=None as input to policy_net. It will return lstm_state and hidden. Discard hidden. Use the last lstm_state as the current Q values
        ### CODE ####
        current_action, hidden = self.policy_net(states, hidden=None)
        idx = actions.unsqueeze(1)
        current_val = current_action.gather(1, idx)
        # Compute Q function of next state
        # Similar to previous, use hidden=None as input to policy_net. And discard the hidden returned by policy_net
        ### CODE ####
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).to(device)
            next_action, hidden = self.policy_net(next_states, hidden=None)
            # Find maximum Q-value of action at next state from policy net
            ### CODE ####
            max_action = next_action.max(1)
            max_val = max_action[0].detach()
        
        # Find maximum Q-value of action at next state from policy net
        # Use the last lstm_state as the Q values of next state
        ### CODE ####
        next_val = mask.float().to(device) * max_val # do not require gradients
        expect_val = next_val * self.discount_factor + rewards
        loss = torch.nn.functional.smooth_l1_loss(current_val, expect_val)
        
        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
