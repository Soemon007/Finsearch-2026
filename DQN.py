import copy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from backtest import backtester

# ----------------------------------------------------------------------------
# Setup & Global Constants
# ----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 7, 123, 2024, 2026]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Engineered stationary features, upgraded with a 50-day macro trend ratio
STATE_FEATURES = [
    'Price_SMA5_Ratio', 'SMA5_SMA20_Ratio', 'SMA20_SMA50_Ratio',
    'Momentum_5', 'Momentum_10', 'Volatility_10', 'RSI_14', 'MACD_Hist_Rel', 'BB_PctB'
]
STATE_DIM = len(STATE_FEATURES) + 1   # + Position flag
ACTION_DIM = 3                        # 0: Sell, 1: Hold, 2: Buy

TRANSACTION_COST = 0.0001
SLIPPAGE = 0.0002

VAL_FRACTION = 0.15   # Last 15% of train split held out for validation checkpointing
VOL_WINDOW = 10       # Window for downside volatility calculation

EPISODES = 150
BATCH_SIZE = 64
TAU = 0.005           # Soft update parameter for target network

STATE_MEAN = None
STATE_STD = None


# ----------------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------------
def build_feature_frame(prices):
    """Turn raw price series into stationary features, including a 50-day trend."""
    df = pd.DataFrame({'Price': prices.astype(np.float64)})
    
    sma5 = df['Price'].rolling(window=5).mean()
    sma20 = df['Price'].rolling(window=20).mean()
    sma50 = df['Price'].rolling(window=50).mean()
    
    df['Price_SMA5_Ratio'] = df['Price'] / sma5 - 1
    df['SMA5_SMA20_Ratio'] = sma5 / sma20 - 1
    df['SMA20_SMA50_Ratio'] = sma20 / sma50 - 1
    
    df['Momentum_5'] = df['Price'].pct_change(periods=5)
    df['Momentum_10'] = df['Price'].pct_change(periods=10)
    df['Volatility_10'] = df['Price'].pct_change().rolling(window=10).std()

    # RSI(14) centered around 0 ([-1, 1] scale)
    delta = df['Price'].diff()
    avg_gain = delta.clip(lower=0).rolling(window=14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df['RSI_14'] = (rsi.fillna(50.0) - 50.0) / 50.0

    # Price-relative MACD Histogram
    ema12 = df['Price'].ewm(span=12, adjust=False).mean()
    ema26 = df['Price'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist_Rel'] = (macd_line - signal_line) / df['Price']

    # Bollinger %B centered around 0
    bb_std = df['Price'].rolling(window=20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    df['BB_PctB'] = ((df['Price'] - bb_lower) / band_width) - 0.5

    return df


def _prepare_features(train_prices, test_prices):
    """Build inner-train / validation / test feature frames without leakage."""
    global STATE_MEAN, STATE_STD

    full_prices = pd.concat([train_prices, test_prices])
    full_prices = full_prices[~full_prices.index.duplicated(keep='first')].sort_index()

    full_features = build_feature_frame(full_prices).dropna()
    for col in STATE_FEATURES:
        full_features[col] = full_features[col].astype(np.float32)

    train_features_df = full_features.loc[full_features.index.isin(train_prices.index)].reset_index(drop=True)
    test_features_df = full_features.loc[full_features.index.isin(test_prices.index)].reset_index(drop=True)

    val_split_idx = int(len(train_features_df) * (1 - VAL_FRACTION))
    inner_train_features_df = train_features_df.iloc[:val_split_idx].reset_index(drop=True)
    val_features_df = train_features_df.iloc[val_split_idx:].reset_index(drop=True)

    print(f"Inner-train rows: {len(inner_train_features_df)} | Validation rows: {len(val_features_df)} "
          f"| Test rows: {len(test_features_df)}")

    train_stats_df = inner_train_features_df[STATE_FEATURES].astype(np.float64)
    STATE_MEAN = train_stats_df.mean().values.astype(np.float32)
    STATE_STD = train_stats_df.std().replace(0, 1e-6).values.astype(np.float32)

    return inner_train_features_df, val_features_df, test_features_df


# ----------------------------------------------------------------------------
# Environment (Fast, Sortino Reward, Whipsaw Penalty, Correct Share Math)
# ----------------------------------------------------------------------------
class StockTradingEnv:
    def __init__(self, dataframe, initial_balance=100000,
                 transaction_cost=TRANSACTION_COST, slippage=SLIPPAGE,
                 vol_window=VOL_WINDOW):
        self.df = dataframe.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.state_features = STATE_FEATURES
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.vol_window = vol_window
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.prev_action = 1  # Default to Hold
        self.net_worth = self.initial_balance
        self.returns_history = deque(maxlen=self.vol_window)
        return self._get_state()

    def _get_state(self):
        raw = self.df.loc[self.current_step, self.state_features].values.astype(np.float32)
        norm = (raw - STATE_MEAN) / STATE_STD
        position_flag = np.float32(1.0 if self.shares_held > 0 else 0.0)
        return np.concatenate([norm.astype(np.float32), [position_flag]]).astype(np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Price']

        # 2: Buy, 0: Sell, 1: Hold
        if action == 2 and self.shares_held == 0:
            if self.balance > 0:
                execution_price = current_price * (1 + self.slippage)
                # Fixed math: divide by (price * fee_multiplier) to prevent accidental cash overdraft
                shares_bought = self.balance / (execution_price * (1 + self.transaction_cost))
                cost = shares_bought * execution_price
                fee = cost * self.transaction_cost
                self.balance -= (cost + fee)
                self.shares_held += shares_bought
                
        elif action == 0 and self.shares_held > 0:
            execution_price = current_price * (1 - self.slippage)
            revenue = self.shares_held * execution_price
            fee = revenue * self.transaction_cost
            self.balance += (revenue - fee)
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = self.df.loc[self.current_step, 'Price']
        new_net_worth = self.balance + (self.shares_held * next_price)

        pct_return = (new_net_worth - self.net_worth) / (self.net_worth + 1e-8)
        self.returns_history.append(pct_return)

        # Sortino scaling: isolate negative returns for downside volatility
        downside_returns = [r for r in self.returns_history if r < 0]
        if len(downside_returns) >= 3:
            downside_vol = float(np.std(downside_returns))
        else:
            downside_vol = 0.01
        downside_vol = max(downside_vol, 1e-4)

        # Whipsaw penalty: discourage erratic flipping between Buy and Sell
        whipsaw_penalty = 0.0
        if action != self.prev_action and action != 1 and self.prev_action != 1:
            whipsaw_penalty = 0.001  
        self.prev_action = action

        # Reward positive breakouts; penalize downside drift and whipsaws
        reward = float(np.clip((pct_return / downside_vol) - whipsaw_penalty, -5.0, 5.0))

        self.net_worth = new_net_worth
        next_state = self._get_state() if not done else None

        return next_state, reward, done


# ----------------------------------------------------------------------------
# Dueling Q-Network (Fast 64-Node Architecture)
# ----------------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Mean-centered aggregation for stability and identifiability
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# ----------------------------------------------------------------------------
# Replay Buffer
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------------
# DQN Agent (Double Dueling with Fast Continuous Soft Updates)
# ----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.02,
                 min_replay_size=1000, hidden_dim=64, tau=TAU):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.min_replay_size = min_replay_size
        self.tau = tau

        self.policy_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values, dim=1).item()

    def train_step(self, batch_size=64):
        if len(self.memory) < max(batch_size, self.min_replay_size):
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN: Policy net selects action, Target net evaluates value
            best_next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states_t).gather(1, best_next_actions)
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Polak-Ruppert continuous soft update (replaces slow episodic hard updates)
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def evaluate_greedy(agent, env):
    """Run one full greedy (epsilon=0) pass and return the ending net worth."""
    state = env.reset()
    done = False
    agent.policy_net.eval()
    with torch.no_grad():
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_t).squeeze(0)

            sell_q = q_values[0].item()
            hold_q = q_values[1].item()
            buy_q  = q_values[2].item()

            THRESHOLD = 0.10

            if buy_q > max(sell_q, hold_q) + THRESHOLD:
                action = 2
            elif sell_q > max(buy_q, hold_q) + THRESHOLD:
                action = 0
            else:
                action = 1
            state, _, done = env.step(action)
    agent.policy_net.train()
    return env.net_worth


def train_one_seed(seed, inner_train_features_df, val_features_df, verbose=True):
    set_seed(seed)

    train_env = StockTradingEnv(inner_train_features_df, initial_balance=100000)
    val_env = StockTradingEnv(val_features_df, initial_balance=100000)

    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    best_val_net_worth = -np.inf
    best_weights = copy.deepcopy(agent.policy_net.state_dict())
    net_worth_history = []

    for episode in range(1, EPISODES + 1):
        state = train_env.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = train_env.step(action)

            if done:
                next_state = state

            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step(BATCH_SIZE)

            state = next_state

        agent.update_epsilon()
        net_worth_history.append(train_env.net_worth)

        val_net_worth = evaluate_greedy(agent, val_env)
        if val_net_worth > best_val_net_worth:
            best_val_net_worth = val_net_worth
            best_weights = copy.deepcopy(agent.policy_net.state_dict())

        if verbose and episode % 10 == 0:
            print(f"  [seed {seed}] Episode {episode:03d}/{EPISODES} | "
                  f"Train net worth: Rs.{train_env.net_worth:,.2f} | "
                  f"Val net worth: Rs.{val_net_worth:,.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    return best_weights, best_val_net_worth, net_worth_history


# ----------------------------------------------------------------------------
# Public Entry Points (Called from main.py)
# ----------------------------------------------------------------------------
def DQN_modelling(train_prices, test_prices):
    """Train D3QN across multiple seeds and backtest best policy on test split."""
    inner_train_features_df, val_features_df, test_features_df = _prepare_features(train_prices, test_prices)

    print("Starting Double Dueling DQN Model Training across seeds", SEEDS, "...")

    seed_results = []
    best_overall = {"val_net_worth": -np.inf}

    for seed in SEEDS:
        print(f"\n=== Training seed {seed} ===")
        weights, val_net_worth, history = train_one_seed(seed, inner_train_features_df, val_features_df)
        seed_results.append((seed, val_net_worth))
        print(f"Seed {seed} finished | Best validation net worth: Rs.{val_net_worth:,.2f}")

        if val_net_worth > best_overall["val_net_worth"]:
            best_overall = {
                "seed": seed,
                "weights": weights,
                "val_net_worth": val_net_worth,
                "history": history,
            }

    val_net_worths = np.array([v for _, v in seed_results], dtype=np.float64)
    print("\nTraining Complete across all seeds!")
    print(f"Validation net worth by seed: {seed_results}")
    print(f"Validation net worth mean: Rs.{val_net_worths.mean():,.2f} | std: Rs.{val_net_worths.std():,.2f}")
    print(f"Selected seed {best_overall['seed']} (best on validation, Rs.{best_overall['val_net_worth']:,.2f}) for final test evaluation.")

    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent.policy_net.load_state_dict(best_overall["weights"])
    agent.policy_net.eval()

    test_env = StockTradingEnv(test_features_df, initial_balance=100000)
    state = test_env.reset()
    done = False
    actions = []

    print("\nRunning Trained Agent on the TEST split to generate actions...")

    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_t).squeeze(0)

            sell_q = q_values[0].item()
            hold_q = q_values[1].item()
            buy_q  = q_values[2].item()

            THRESHOLD = 0.10

            if buy_q > max(sell_q, hold_q) + THRESHOLD:
                action = 2
            elif sell_q > max(buy_q, hold_q) + THRESHOLD:
                action = 0
            else:
                action = 1

        actions.append(action)
        state, reward, done = test_env.step(action)

    print(f"Collected {len(actions)} actions successfully!")
    print(f"Ending Test Portfolio Net Worth (internal env, for reference): Rs.{test_env.net_worth:,.2f}")

    mapping = {2: 1, 1: 0, 0: -1}
    signals = [mapping[a] for a in actions]

    # Uses your standalone backtest.py defaults untouched
    dqn_portfolio = backtester(signals, test_prices)

    return dqn_portfolio, best_overall, seed_results


def DQN_summary(dqn_portfolio, best_overall=None, seed_results=None):
    """Print, plot, and save D3QN results."""
    print(dqn_portfolio.head())

    if seed_results is not None:
        print(f"Validation net worth by seed: {seed_results}")

    if best_overall is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(best_overall["history"],
                  label=f"D3QN Portfolio Value (Train, seed {best_overall['seed']})", color="green")
        plt.axhline(y=100000, color="r", linestyle="--", label="Initial Balance (Rs.100,000)")
        plt.title("Portfolio Net Worth Progress Across Training Episodes (Winning Seed)")
        plt.xlabel("Episodes")
        plt.ylabel("Portfolio Value (Rs.)")
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(dqn_portfolio["PortfolioValue"])
    plt.title("Double Dueling DQN (D3QN) Portfolio Value")
    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

    dqn_portfolio.to_csv("dqn_portfolio.csv", index=False)
    print("\nSaved backtested results to dqn_portfolio.csv")