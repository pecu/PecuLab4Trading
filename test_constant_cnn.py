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

if __name__ == '__main__':
    window_size = 10
    max_bid=3 

    print('CUDA is available: {}'.format(torch.cuda.is_available()))
    print('CUDA current device: {}'.format(torch.cuda.current_device()))
    print('CUDA current addr: {}'.format(torch.cuda.device(0)))
    print('CUDA device count: {}'.format(torch.cuda.device_count()))
    print('CUDA device name: {}'.format(torch.cuda.get_device_name(0)))

    print(device)
    
    # data_path='data/TWII_2019_testing.csv'
    #data_path='data/SPY_2019_testing.csv'
    #data_path='data/ETH_2019_testing.csv'
    #data_path='data/GLD_2019_testing.csv'
    #data_path='data/BNO_2019_testing.csv'
    #data_path='data/SLV_2019_testing.csv'
    #data_path='data/GOOG_2019_testing.csv'
    #data_path='data/AAPL_2019_testing.csv'
    #data_path='data/EWJ_2019_testing.csv'
    #data_path='data/ACWI_2019_testing.csv'

    # YC
    # data_path='data/2330_kbar_daily_gmt_ohlc.csv'
    data_path='data/1229_kbar_daily_gmt_ohlc.csv'


    asset=data_path.split("/")[1].split("_")[0]
    data_path2=data_path.split("/")[1].split(".")[0]

    date='2021_04_13_11_45'#ETH BEST
    #date='2021_04_16_14_56' #sortino

    big_random_list=[]
    big_train=[]
    big_no_train=[]
    big_random_list_r=[]
    big_train_r=[]
    big_no_train_r=[]


    Weight_PATH=f'model/{date}/PPO_ETH3000_2020_1000_256_cnn.pth'#ETH BEST
    #Weight_PATH=f'model/{date}/PPO_ETH3000_2020_150_256_cnn.pth' #sortino

    hidden=256
    for _ in tqdm(range(5)):
        for pre_train,randomm in [(True,False),(False,False),(False,True)]: 

            process_list, order,data,signal,re_list,so_list=test()

            if randomm:
                random_list=process_list.copy()
                random_list_r=re_list.copy()
            else:
                if not pre_train:
                    no_train=process_list.copy()
                    no_train_r=re_list.copy()
                elif pre_train:
                    train=process_list.copy()
                    train_r=re_list.copy()

        big_random_list.append(random_list)
        big_random_list_r.append(random_list_r)
        big_train.append(train)
        big_train_r.append(train_r)
        big_no_train.append(no_train)
        big_no_train_r.append(no_train_r)

    #####################plot

    brl=np.array(big_random_list)
    bt=np.array(big_train)
    bnt=np.array(big_no_train)

    brl50=np.percentile(brl, 50, axis=0)
    brl95=np.percentile(brl, 95, axis=0)
    brl5=np.percentile(brl, 5, axis=0)

    bt50=np.percentile(bt, 50, axis=0)
    bt95=np.percentile(bt, 95, axis=0)
    bt5=np.percentile(bt, 5, axis=0)

    bnt50=np.percentile(bnt, 50, axis=0)
    bnt95=np.percentile(bnt, 95, axis=0)
    bnt5=np.percentile(bnt, 5, axis=0)

    pd.Series(brl50).plot(label='random',legend=True)
    plt.fill_between(range(len(random_list)),brl5, brl95, alpha=0.25)


    pd.Series(bnt50).plot(label='no_ppo',legend=True)
    plt.fill_between(range(len(no_train)),bnt5, bnt95, alpha=0.25)

    pd.Series(bt50).plot(label='ppo',legend=True)
    plt.fill_between(range(len(train)),bt5, bt95, alpha=0.25)

    pd.Series(data['Close'][window_size:].reset_index(drop=True)-data['Close'].iloc[window_size]).plot(label='index',legend=True)
    plt.savefig(f'fig/ppo/{date}/PPO_{data_path2}_{max_bid}_s{random_seed}_index_test.png',dpi=150)
    plt.title(f"testing_{data_path2}_{max_bid}_s{random_seed}")

#once test log
###once test setting
max_bid=3
pre_train=True
randomm=False
process_list, order,data,signal ,re_list,so_list=test()
###

index_buy=[]
mark_buy=[]
index_sell=[]
mark_sell=[]

shift2=[9]*(window_size)
shift2.extend(signal)
len(shift2)
# signal
close_sell_id=[]
close_buy_id=[]
close_sell=[]
close_buy=[]
long_short=0

for i,sig in enumerate(shift2):
    if long_short>0 and sig==0:
        close_sell_id.append(i)
    if long_short<0 and sig==2:
        close_buy_id.append(i)
    if sig==0:
        if -max_bid<long_short<=max_bid:
            long_short-=1
    elif sig==1:
        pass
    elif sig==2:
        if -max_bid<=long_short<max_bid:
            long_short+=1

for mark in order:
    if mark['type']=='buy':
        index_buy.append(mark['id']-1)
        mark_buy.append(mark['price'])
        if mark['close']:
            close_sell.append(mark['price']+mark['gain'])
    elif mark['type']=='sell':
        index_sell.append(mark['id']-1)
        mark_sell.append(mark['price'])
        if mark['close']:
            close_buy.append(mark['price']+mark['gain'])

fig, (ax0, ax1)= plt.subplots(2, 1,dpi=150, sharex=True)
ax0.set_title(f'{asset}')
ax1.set_xlabel('time(days)')
ax0.plot(data['Close'])
ax0.scatter(index_buy,mark_buy,color='red', marker='^',linewidths=1)
ax0.scatter(close_buy_id,close_buy,color='red', marker='^',linewidths=1)
ax0.scatter(index_sell,mark_sell,color='green', marker='v',linewidths=1)
ax0.scatter(close_sell_id,close_sell,color='green', marker='v',linewidths=1)
ax1.set_title('acc profit')
shift=[0]*(window_size)
shift.extend(process_list)
ax1.plot(shift)
plt.savefig(f'fig/ppo/{date}/PPO_{data_path2}_{max_bid}_cmp_test.png',dpi=150)