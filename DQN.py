"""
DQN.py
------
Vanilla Deep Q-Network trading agent (single online net + single target net,
plain MLP, no dueling / prioritized replay / double-DQN / anything
non-standard).

MODULE STRUCTURE (matches ARIMA.py's pattern for main.py to call)
====================================================================
- DQN_modelling(train_prices, test_prices) -> trains across seeds, evaluates
  the best-on-validation seed on the test split, and returns the backtested
  portfolio plus some diagnostics.
- DQN_summary(dqn_portfolio, seed_results) -> prints, plots, and saves the
  results, mirroring ARIMA_summary.

This file does NOT import from main.py or feature_engineering.py's module
scope. train_prices/test_prices are passed in as function arguments by
main.py, the same way ARIMA_modelling/Forecast receive train_series/
test_series/test_prices. Anything defined only inside another function
(e.g. inside main()) can't be imported by name -- it's a local variable that
stops existing the moment that function returns -- so this module is
structured to never rely on that.

WHAT WAS WRONG BEFORE, AND WHAT CHANGED (kept for context)
=============================================================
1. TRAIN/TEST LEAKAGE + MISALIGNED EVALUATION -- fixed by training only on
   the train split and generating test-split actions via a separate greedy
   (epsilon=0) pass, matching how ARIMA's `Forecast` scores against
   `test_prices`.
2. UNNORMALISED, MIXED-SCALE STATE -- z-score normalised using stats computed
   on the INNER-train slice only (see item 8).
3. RAW-RUPEE REWARD -> replaced with a risk-adjusted percentage return (item 7).
4. LOSS/OPTIMISATION STABILITY -- Huber loss + gradient clipping. Still a
   single, plain feed-forward Q-network -- no dueling/prioritized
   replay/double-DQN, deliberately kept vanilla.
5. TRAINING SCHEDULE -- more episodes, tuned epsilon decay so the greedy
   policy used for evaluation is actually exercised during training.
6. TRAIN/EVAL COST MISMATCH -- `StockTradingEnv.step()` now charges the same
   fee/slippage as `backtester()`, so training reward already reflects the
   costs the agent is scored on.
7. RISK-ADJUSTED REWARD SHAPING -- reward is daily % return divided by recent
   realised volatility of the portfolio's own returns (differential-Sharpe
   style), clipped for stability.
8. VALIDATION-BASED CHECKPOINTING -- the train split is further divided into
   an inner-train slice and a held-out validation slice (last 15%,
   chronological); checkpointing and normalisation stats are based on that
   split, not on training-set noise.
9. STATIONARY STATE FEATURES -- raw price levels replaced with scale-free
   ratios (Price/SMA_5 - 1, SMA_5/SMA_20 - 1); volatility computed on returns.
10. POSITION AWARENESS -- state includes a flag for whether the agent
    currently holds a position.
11. MULTIPLE SEEDS, SELECTED ON VALIDATION -- trains across several seeds;
    the final model is the seed with the best VALIDATION net worth (never
    training or test), with mean/std across seeds reported for transparency.
"""

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
# Setup
# ----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 7, 123, 2024, 2026]   # multiple seeds -> mean/std + best-on-validation selection


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Engineered (stationary) features the network actually sees.
STATE_FEATURES = ['Price_SMA5_Ratio', 'SMA5_SMA20_Ratio', 'Momentum_5', 'Momentum_10', 'Volatility_10']
STATE_DIM = len(STATE_FEATURES) + 1   # + position flag
ACTION_DIM = 3                         # 0: Sell, 1: Hold, 2: Buy

# Same cost structure as backtest.py's defaults -- kept as named constants so
# the environment and the backtester can never silently drift apart.
TRANSACTION_COST = 0.0001
SLIPPAGE = 0.0002

VAL_FRACTION = 0.15   # last 15% of the train split, chronologically, held out for checkpointing
VOL_WINDOW = 10        # window for the risk-adjusted reward's volatility estimate

EPISODES = 150
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 5

# Set by _prepare_features() before any StockTradingEnv is created; read by
# StockTradingEnv._get_state(). Kept as module state (like STATE_FEATURES)
# rather than threading it through every function call.
STATE_MEAN = None
STATE_STD = None


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def build_feature_frame(prices):
    """Turn a raw price series into the engineered, stationary feature set.

    Price levels (SMA_5, SMA_20, Price itself) are converted to ratios so the
    network is never fed a raw, non-stationary price level -- only scale-free
    quantities that should look similar whether the price is near its
    training range or has drifted.
    """
    df = pd.DataFrame({'Price': prices.astype(np.float64)})
    sma5 = df['Price'].rolling(window=5).mean()
    sma20 = df['Price'].rolling(window=20).mean()
    df['Price_SMA5_Ratio'] = df['Price'] / sma5 - 1
    df['SMA5_SMA20_Ratio'] = sma5 / sma20 - 1
    df['Momentum_5'] = df['Price'].pct_change(periods=5)
    df['Momentum_10'] = df['Price'].pct_change(periods=10)
    df['Volatility_10'] = df['Price'].pct_change().rolling(window=10).std()
    return df


def _prepare_features(train_prices, test_prices):
    """Build inner-train / validation / test feature frames from raw prices.

    Features are computed on the CONTINUOUS price series (train followed by
    test) so the first rows of the test split still have a valid rolling
    window (looking back into the tail of the training data -- legitimate,
    since a live trading system would have that history available too).

    The train split is further divided (chronologically, last VAL_FRACTION)
    into an inner-train slice and a held-out validation slice used only for
    checkpointing. Normalisation stats are computed on the inner-train slice
    only -- no lookahead into validation or test.
    """
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
# Environment (all-in/all-out Sell/Hold/Buy, next-day fill, cost-aware,
# risk-adjusted reward, position-aware state)
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
        self.net_worth = self.initial_balance
        self.returns_history = deque(maxlen=self.vol_window)
        return self._get_state()

    def _get_state(self):
        raw = self.df.loc[self.current_step, self.state_features].values.astype(np.float32)
        norm = (raw - STATE_MEAN) / STATE_STD
        # Position flag: is the agent currently holding shares? Lets it learn
        # "enter fresh" and "hold through the cost of flipping" as separate
        # decisions instead of re-deriving position implicitly.
        position_flag = np.float32(1.0 if self.shares_held > 0 else 0.0)
        return np.concatenate([norm.astype(np.float32), [position_flag]]).astype(np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Price']

        # 0: Sell, 1: Hold, 2: Buy
        # Execution price, fee, and slippage mirror backtester.py's BUY/SELL
        # blocks exactly, so the reward the agent trains on already prices in
        # the same costs it will be scored on later.
        if action == 2:
            if self.balance > 0:
                execution_price = current_price * (1 + self.slippage)
                shares_bought = (self.balance / execution_price) * (1 + self.transaction_cost)
                cost = shares_bought * execution_price
                fee = cost * self.transaction_cost
                self.balance -= (cost + fee)
                self.shares_held += shares_bought
        elif action == 0:
            if self.shares_held > 0:
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

        # Risk-adjusted reward: return scaled by the recent realised
        # volatility of the portfolio's own daily returns (a lightweight
        # differential-Sharpe-style signal), so the agent is pushed toward
        # steady gains rather than volatile ones. Falls back to a fixed prior
        # vol until there's enough history to estimate it, and is clipped for
        # numerical stability.
        if len(self.returns_history) >= 5:
            vol = float(np.std(self.returns_history))
        else:
            vol = 0.01  # reasonable daily-return-vol prior before we have data
        vol = max(vol, 1e-4)
        reward = float(np.clip(pct_return / vol, -5.0, 5.0))

        self.net_worth = new_net_worth
        next_state = self._get_state() if not done else None

        return next_state, reward, done


# ----------------------------------------------------------------------------
# Q-Network (plain vanilla MLP)
# ----------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ----------------------------------------------------------------------------
# Replay buffer
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
# DQN Agent (vanilla: single online net + single target net, hard updates)
# ----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.97, epsilon_min=0.02,
                 min_replay_size=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.min_replay_size = min_replay_size

        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss: more stable than MSE
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
            max_next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def evaluate_greedy(agent, env):
    """Run one full greedy (epsilon=0) pass and return the ending net worth."""
    state = env.reset()
    done = False
    agent.policy_net.eval()
    with torch.no_grad():
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_t)
            action = torch.argmax(q_values, dim=1).item()
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

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        net_worth_history.append(train_env.net_worth)

        # Checkpoint on a greedy pass over the held-out VALIDATION split, not
        # on training-set net worth -- avoids keeping the episode that just
        # got lucky on training noise.
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
# Public entry points, called from main.py
# ----------------------------------------------------------------------------
def DQN_modelling(train_prices, test_prices):
    """Train a vanilla DQN across multiple seeds and backtest the best
    (validation-selected) policy on the test split.

    Parameters
    ----------
    train_prices, test_prices : pandas Series
        Same train/test price splits produced by feature_engineering() and
        already used by ARIMA_modelling/Forecast.

    Returns
    -------
    dqn_portfolio : DataFrame
        Backtested portfolio history (from backtester()).
    best_overall : dict
        Diagnostics about the winning seed: {"seed", "weights",
        "val_net_worth", "history"}.
    seed_results : list of (seed, val_net_worth)
        Per-seed validation performance, for the mean/std transparency check.
    """
    inner_train_features_df, val_features_df, test_features_df = _prepare_features(train_prices, test_prices)

    print("Starting DQN Model Training across seeds", SEEDS, "...")

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
    print(f"Validation net worth mean: Rs.{val_net_worths.mean():,.2f} | "
          f"std: Rs.{val_net_worths.std():,.2f}")
    print(f"Selected seed {best_overall['seed']} (best on validation, "
          f"Rs.{best_overall['val_net_worth']:,.2f}) for final test evaluation.")

    # Rebuild the winning agent and load its best (validation-selected) weights.
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent.policy_net.load_state_dict(best_overall["weights"])
    agent.policy_net.eval()

    # Evaluate on the TEST split (greedy policy, epsilon=0) -- this is what
    # actually gets scored against ARIMA via the backtester.
    test_env = StockTradingEnv(test_features_df, initial_balance=100000)
    state = test_env.reset()
    done = False
    actions = []

    print("\nRunning Trained Agent on the TEST split to generate actions...")

    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_t)
        action = torch.argmax(q_values, dim=1).item()

        actions.append(action)
        state, reward, done = test_env.step(action)

    print(f"Collected {len(actions)} actions successfully!")
    print(f"Ending Test Portfolio Net Worth (internal env, for reference): Rs.{test_env.net_worth:,.2f}")

    mapping = {
        2: 1,    # BUY
        1: 0,    # HOLD
        0: -1,   # SELL
    }
    signals = [mapping[a] for a in actions]

    dqn_portfolio = backtester(signals, test_prices)

    return dqn_portfolio, best_overall, seed_results


def DQN_summary(dqn_portfolio, best_overall=None, seed_results=None):
    """Print, plot, and save DQN results -- mirrors ARIMA_summary."""
    print(dqn_portfolio.head())

    if seed_results is not None:
        print(f"Validation net worth by seed: {seed_results}")

    if best_overall is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(best_overall["history"],
                  label=f"DQN Portfolio Value (Train, seed {best_overall['seed']})", color="green")
        plt.axhline(y=100000, color="r", linestyle="--", label="Initial Balance (Rs.100,000)")
        plt.title("Portfolio Net Worth Progress Across Training Episodes (Winning Seed)")
        plt.xlabel("Episodes")
        plt.ylabel("Portfolio Value (Rs.)")
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(dqn_portfolio["PortfolioValue"])
    plt.title("DQN Portfolio Value")
    plt.xlabel("Trading Day")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

    dqn_portfolio.to_csv("dqn_portfolio.csv", index=False)
    print("\nSaved backtested results to dqn_portfolio.csv")