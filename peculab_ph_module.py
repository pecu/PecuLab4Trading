import talib
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
from Modules_cnn import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden = 256
pre_train = True
Weight_PATH = None
window_size = 10
max_bid=3 
pre_train=True
randomm=False
date = None
asset = None
data_path = None
data_path2 = None

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = state.float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def test():
    global random_seed

    state_dim = 10
    action_dim = 3
    max_timesteps = 100
    n_latent_var = int(hidden)         # number of variables in hidden layer
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    N = 255 
    rf =0.0016 
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    if pre_train:
        try:
            ppo.policy_old.load_state_dict(torch.load(Weight_PATH))
            ppo.policy.load_state_dict(torch.load(Weight_PATH))
        except:
            ppo.policy_old.load_state_dict(torch.load(Weight_PATH,map_location=torch.device('cpu')))
            ppo.policy.load_state_dict(torch.load(Weight_PATH,map_location=torch.device('cpu')))

    ppo.policy_old = ppo.policy_old.to(device)
    ppo.policy = ppo.policy.to(device)
    ppo.policy.eval()
    ppo.policy_old.eval()

    model_fn = 'baseline_95.40__candlestick_500.pt'
    cnn = CNN().to(device)

    if pre_train:
        try:
            cnn.load_state_dict(torch.load(model_fn))
        except:
            cnn.load_state_dict(torch.load(
                model_fn, map_location=torch.device('cpu')))

    cnn.eval()


    raw_data = pd.read_csv(data_path)
    raw_data['Gmt time'] = pd.to_datetime(raw_data['Gmt time'], dayfirst=True)
    max_timesteps = len(raw_data)-window_size

    
    
    process_list = []
    order = []
    long_short = 0
    order_count = 0
    signal=[]
    last_ungain=0
    re_list=[]
    sortino=0
    so_list=[]

    for t in range(max_timesteps):

        data = raw_data[['Open', 'High', 'Low', 'Close']]
        # data = raw_data[['Open', 'High', 'Low', 'Close','slowk','slowd','macdhist','RSI','WILLR']]

        gaf = state_convert(raw_data[['Open', 'High', 'Low', 'Close']].iloc[t:t+window_size])
        state = cnn(gaf)
        state = torch.cat((torch.tensor([long_short, order_count]).unsqueeze(0).to(device), state), axis=1)


        # Running policy_old:
        if not randomm:
            action = ppo.policy_old.act(state, memory)
        elif randomm:
            #random.seed(1276)
            action = random.randint(0, 2)


        signal.append(action)
        state, reward, order,done, long_short, order_count,last_ungain = market_step(data, action, t+1+window_size, order, window_size,max_bid,last_ungain,sortino)
        
        re_list.append(reward/data['Close'].iloc[t+window_size])
        if len(re_list)>10:
            sortino = sortino_ratio(pd.Series(re_list),N,rf)
        else:
            sortino=0
        so_list.append(sortino)
        process_list.append(last_ungain)
        

    return process_list, order , data ,signal, re_list,so_list

def sortino_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    return_series_d=return_series.copy()
    return_series_d[return_series_d>=rf]=rf
    sigma = return_series_d.std() * np.sqrt(N)
    return mean / sigma