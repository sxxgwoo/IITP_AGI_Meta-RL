import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import os

# -------------------------
# 5x5 GridWorld Environment
# -------------------------


class GridWorld5x5:
    """
    Simple 5x5 grid world.
    - Start at (0, 0)
    - Goal at (4, 4)
    - Actions: 0=up, 1=down, 2=left, 3=right
    - Reward: -1 per step, +10 on reaching goal
    - Episode terminates at goal or after max_steps
    """

    def __init__(self, max_steps=20, seed=0):
        self.size = 5
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.max_steps = max_steps

        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pos = list(self.start_pos)
        self.t = 0
        return self._pos_to_state()

    def _pos_to_state(self):
        # flatten (x,y) to index 0..24
        x, y = self.pos
        return x * self.size + y

    def step(self, action):
        self.t += 1
        x, y = self.pos

        if action == 0:  # up
            x = max(x - 1, 0)
        elif action == 1:  # down
            x = min(x + 1, self.size - 1)
        elif action == 2:  # left
            y = max(y - 1, 0)
        elif action == 3:  # right
            y = min(y + 1, self.size - 1)
        else:
            raise ValueError("Invalid action")

        self.pos = [x, y]
        done = False
        reward = -0.1

        if (x, y) == self.goal_pos:
            reward = 1.0
            done = True

        if self.t >= self.max_steps:
            done = True

        return self._pos_to_state(), reward, done, {}

    @property
    def num_states(self):
        return self.size * self.size

    @property
    def num_actions(self):
        return 4


# -------------------------
# Offline Dataset Collector
# -------------------------


def collect_random_dataset(env, num_episodes=2000, max_steps=20, seed=0):
    """
    Collect offline data using a random policy.
    Returns tensors: states, actions, rewards, next_states, dones
    """
    rng = random.Random(seed)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for _ in range(num_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = rng.randint(0, env.num_actions - 1)
            s_next, r, done, _ = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            dones.append(float(done))

            s = s_next
            if done:
                break

    states = torch.tensor(states, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.long)
    dones = torch.tensor(dones, dtype=torch.float32)

    return states, actions, rewards, next_states, dones


# -------------------------
# IQL Networks
# -------------------------


class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_states, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_idx):
        """
        state_idx: LongTensor [B], discrete state index 0..num_states-1
        """
        x = self.embedding(state_idx)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.out(x)
        return q


class VNetwork(nn.Module):
    def __init__(self, num_states, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_states, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state_idx):
        x = self.embedding(state_idx)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.out(x)
        return v.squeeze(-1)  # [B]


class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_states, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_idx):
        x = self.embedding(state_idx)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits  # [B, num_actions]

    def act(self, state_idx, greedy=True):
        with torch.no_grad():
            logits = self.forward(state_idx.unsqueeze(0))  # [1, A]
            probs = F.softmax(logits, dim=-1)
            if greedy:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.distributions.Categorical(probs).sample()
        return int(action.item())


# -------------------------
# IQL Training Loop
# -------------------------


@dataclass
class IQLConfig:
    gamma: float = 0.99
    tau: float = 0.7  # expectile parameter
    beta: float = 3.0  # temperature for advantage weighting
    batch_size: int = 128
    num_epochs: int = 50
    lr_q: float = 3e-4
    lr_v: float = 3e-4
    lr_pi: float = 3e-4
    adv_clip: float = 10.0  # avoid exploding exp(beta * A)


def expectile_loss(diff, tau):
    """
    diff = Q(s,a).detach() - V(s)
    tau in (0, 1)
    Implementation of expectile regression loss.
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


def train_iql(
    env, states, actions, rewards, next_states, dones, cfg=IQLConfig(), device="cpu"
):
    num_states = env.num_states
    num_actions = env.num_actions

    dataset = TensorDataset(states, actions, rewards, next_states, dones)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    q_net = QNetwork(num_states, num_actions).to(device)
    v_net = VNetwork(num_states).to(device)
    pi_net = PolicyNetwork(num_states, num_actions).to(device)

    opt_q = torch.optim.Adam(q_net.parameters(), lr=cfg.lr_q)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=cfg.lr_v)
    opt_pi = torch.optim.Adam(pi_net.parameters(), lr=cfg.lr_pi)
    
    
    avg_return = []
    len_term = 10
    for term in range(len_term):
        for epoch in range(cfg.num_epochs):
            q_losses = []
            v_losses = []
            pi_losses = []

            for batch in dataloader:
                s, a, r, s_next, d = [x.to(device) for x in batch]

                # --------- Q update ---------
                q_values = q_net(s)  # [B, A]
                q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # [B]

                with torch.no_grad():
                    v_next = v_net(s_next)  # [B]
                    target_q = r + cfg.gamma * (1.0 - d) * v_next  # [B]

                q_loss = F.mse_loss(q_sa, target_q)

                opt_q.zero_grad()
                q_loss.backward()
                opt_q.step()

                # --------- V update (expectile regression) ---------
                with torch.no_grad():
                    q_sa_detached = q_sa.detach()  # from previous forward pass

                v_s = v_net(s)  # [B]
                diff = q_sa_detached - v_s
                v_loss = expectile_loss(diff, cfg.tau)

                opt_v.zero_grad()
                v_loss.backward()
                opt_v.step()

                # --------- Policy update (advantage-weighted BC) ---------
                with torch.no_grad():
                    v_s_detached = v_net(s).detach()  # recompute detached V(s)
                    adv = q_sa_detached - v_s_detached
                    adv_clipped = torch.clamp(adv, -cfg.adv_clip, cfg.adv_clip)
                    weights = torch.exp(cfg.beta * adv_clipped)
                    weights = torch.clamp(weights, max=100.0)

                logits = pi_net(s)  # [B, A]
                log_probs = F.log_softmax(logits, dim=-1)
                log_pi_a = log_probs.gather(1, a.unsqueeze(1)).squeeze(1)

                pi_loss = -(weights * log_pi_a).mean()

                opt_pi.zero_grad()
                pi_loss.backward()
                opt_pi.step()

                q_losses.append(q_loss.item())
                v_losses.append(v_loss.item())
                pi_losses.append(pi_loss.item())

            print(
                f"Epoch {epoch+1}/{cfg.num_epochs} | "
                f"Q loss: {np.mean(q_losses):.4f} | "
                f"V loss: {np.mean(v_losses):.4f} | "
                f"Pi loss: {np.mean(pi_losses):.4f}"
            )


        # 3) Evaluate learned policy in the environment
        print("Evaluating learned policy...")
        avg_return.append(evaluate_policy(env, pi_net, device=device, n_episodes=30, greedy=True)) 
    
        plt.figure()
        plt.plot(range(1,term+2),avg_return)
        plt.savefig("wow.png")
    
    
    return q_net, v_net, pi_net


# -------------------------
# Evaluation helper
# -------------------------


def evaluate_policy(env, pi_net, device, n_episodes=20, greedy=True):
    returns = []
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        while not done:
            s_tensor = torch.tensor(s, dtype=torch.long, device=device)
            a = pi_net.act(s_tensor, greedy=greedy)
            s, r, done, _ = env.step(a)
            total_r += r
        returns.append(total_r)
    avg_return = sum(returns) / len(returns)
    print(f"Average return over {n_episodes} episodes: {avg_return:.2f}")
    return avg_return


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = GridWorld5x5(max_steps=20, seed=42)

    # 1) Collect offline dataset with random policy
    print("Collecting offline dataset (random policy)...")
    states, actions, rewards, next_states, dones = collect_random_dataset(
        env, num_episodes=3000, max_steps=20, seed=42
    )

    # 2) Train IQL on the offline dataset
    cfg = IQLConfig(
        gamma=0.99,
        tau=0.7,
        beta=3.0,
        batch_size=128,
        num_epochs=5,
        lr_q=3e-4,
        lr_v=3e-4,
        lr_pi=3e-4,
    )

    train_iql(
        env,
        states,
        actions,
        rewards,
        next_states,
        dones,
        cfg=cfg,
        device=device,
    )

    