import warnings

warnings.filterwarnings('ignore')

# %%


from pathlib import Path
from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import gym
from gym.envs.registration import register
keras = tf.keras

# %% md

### Settings

# %%

np.random.seed(42)
tf.random.set_seed(42)

# %%

sns.set_style('whitegrid')

# %%

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

# %%

results_path = Path('results', 'trading_bot')
if not results_path.exists():
    results_path.mkdir(parents=True)


# %% md

### Helper functions

# %%

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


# %% md

## Set up Gym Environment



trading_days = 252

# %%

register(
    id='trading-v0',
    entry_point='trading_env:TradingEnvironment',
    max_episode_steps=trading_days
)

# %% md

### Initialize Trading Environment



trading_cost_bps = 1e-3
time_cost_bps = 1e-4

# %%

f'Trading costs: {trading_cost_bps:.2%} | Time costs: {time_cost_bps:.2%}'

# %%

trading_environment = gym.make('trading-v0')
trading_environment.env.trading_days = trading_days
trading_environment.env.trading_cost_bps = trading_cost_bps
trading_environment.env.time_cost_bps = time_cost_bps
trading_environment.env.ticker = 'AAPL'
trading_environment.seed(42)

# %% md

### Get Environment Params

# %%

state_dim = trading_environment.observation_space.shape[0]
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps


# %% md

## Define Trading Agent

# %%

class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 load_path,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size):
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        # 之所以用有限长队列，是为了更新缓存，清理无效的欠学习数据
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        # DQN 的两个网络  在文档中搜索 double Q learning 里面讲述了 两个网络的作用
        self.online_network = self.build_model(load_path=load_path)
        self.target_network = self.build_model(trainable=False,load_path=load_path)
        self.update_target()
        
        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []
        
        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []
        
        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True
    def save_model(self,save_path):
        self.online_network.save(save_path)
    
    def build_model(self, trainable=True,load_path=None):
        if load_path is not None:
            return keras.models.load_model("../ckpt")
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.num_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())
    
    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()
    
    def memorize_transition(self, s, a, r, s_prime, not_done):
        # 保存路径到 池子 中
        if not_done:
            self.episode_reward += r  # 累积 收集的r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay
            
            self.episodes += 1
            self.rewards_history.append(self.episode_reward)  # 本episode收集了多少r
            self.steps_per_episode.append(self.episode_length)  # 本 episode 有多长
            self.episode_reward, self.episode_length = 0, 0
        
        self.experience.append((s, a, r, s_prime, not_done))
    
    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        # 回放不需要采样，存储才需要采样，因此不涉及 epsilon
        states, actions, rewards, next_states, not_done = minibatch  # epsilon 算出的 action
        # DQN 的两个网络
        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)
        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))
        
        targets = rewards + not_done * self.gamma * target_q_values
        
        q_values = self.online_network.predict_on_batch(states)
        # actions 是用 epsilon 选出的
        q_values[[self.idx, actions]] = targets  # [B,A]花式索引赋值：只有本次的批采样出的 action 才会被更新
        # online_network(Q-network) 输入是state维度的，输出是 action 维度的
        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)
        
        if self.total_steps % self.tau == 0:  #为了增加稳定性，不会每轮更新
            self.update_target()


# %% md

## Define hyperparameters

# %%

gamma = .99,  # discount factor
tau = 100  # target network update frequency

# %% md

### NN Architecture

# %%
# load_path='../ckpt'
load_path=None
architecture = (256, 256)  # units per layer
learning_rate = 0.0001  # learning rate
l2_reg = 1e-6  # L2 regularization

# %% md

### Experience Replay

# %%

replay_capacity = int(1e6)
batch_size = 4096

# %% md

### $\epsilon$-greedy Policy

# %%

epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_steps = 250
epsilon_exponential_decay = .99

# %% md

## Create DDQN Agent



tf.keras.backend.clear_session()

# %%

ddqn = DDQNAgent(state_dim=state_dim,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 epsilon_exponential_decay=epsilon_exponential_decay,
                 replay_capacity=replay_capacity,
                 load_path=load_path,
                 architecture=architecture,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)

# %%

ddqn.online_network.summary()

# %% md

## Run Experiment

# %% md

### Set parameters

# %%

total_steps = 0
# max_episodes = 1000
max_episodes = 2000

# %% md

### Initialize variables

# %%
# 所有 episode 的综合统计
episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []


# %% md

## Visualization

# %%

def track_results(episode, nav_ma_100, nav_ma_10,
                  market_nav_100, market_nav_10,
                  win_ratio, total, epsilon):
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)
    
    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(episode, format_time(total),
                          nav_ma_100 - 1, nav_ma_10 - 1,
                          market_nav_100 - 1, market_nav_10 - 1,
                          win_ratio, epsilon))


# %% md

## Train Agent

# %%

start = time()
results = []
for episode in range(1, max_episodes + 1):
    this_state = trading_environment.reset()  # 每个指标关于时间独立标准化
    for episode_step in range(max_episode_steps): # 252
        # 步长累加
        action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
        # next_state 与 action 无关，reward 与 前一个step的 action有关
        next_state, reward, done, _ = trading_environment.step(action)
        # 保存路径到 池子 中
        ddqn.memorize_transition(this_state,
                                 action,
                                 reward,
                                 next_state,
                                 0.0 if done else 1.0)
        if ddqn.train:
            ddqn.experience_replay() # 回放数据训练
        if done:
            break
        this_state = next_state
    
    # get DataFrame with seqence of actions, returns and nav values
    result = trading_environment.env.simulator.result()
    
    # get results of last step
    final = result.iloc[-1]
    
    # apply return (net of cost) of last action to last starting nav
    nav = final.nav * (1 + final.strategy_return)
    navs.append(nav)
    
    # market nav
    market_nav = final.market_nav
    market_navs.append(market_nav)
    
    # track difference between agent and market NAV results
    diff = nav - market_nav
    diffs.append(diff)
    
    if episode % 10 == 0:
        track_results(episode,
                      # show mov. average results for 100 (10) periods
                      np.mean(navs[-100:]),
                      np.mean(navs[-10:]),
                      np.mean(market_navs[-100:]),
                      np.mean(market_navs[-10:]),
                      # share of agent wins, defined as higher ending nav
                      np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
                      time() - start, ddqn.epsilon)
        ddqn.save_model('../ckpt')
    if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        # 连续25个交易日跑赢死拿，就结束
        print(result.tail())
        break

trading_environment.close()

# %% md

### Store Results

# %%

results = pd.DataFrame({'Episode': list(range(1, episode + 1)),
                        'Agent': navs,
                        'Market': market_navs,
                        'Difference': diffs}).set_index('Episode')

results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
results.info()

# %%

results.to_csv(results_path / 'results.csv', index=False)

# %%

with sns.axes_style('white'):
    sns.distplot(results.Difference)
    sns.despine()

# %% md

### Evaluate Results

# %%

results.info()


fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

df1 = (results[['Agent', 'Market']]
       .sub(1)
       .rolling(100)
       .mean())
df1.plot(ax=axes[0],
         title='Annual Returns (Moving Average)',
         lw=1)

df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
df2.plot(ax=axes[1],
         title='Agent Outperformance (%, Moving Average)')

for ax in axes:
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
axes[1].axhline(.5, ls='--', c='k', lw=1)

sns.despine()
fig.tight_layout()
fig.savefig(results_path / 'performance', dpi=300)
