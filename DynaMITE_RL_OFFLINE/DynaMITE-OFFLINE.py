import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch.distributions import Categorical, Normal
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
import datetime as dt

import pickle

from envs import GridWorldAlternate
from models import HLT_DynaMITE_VAE

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    steps_per_rollout: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 256
    total_env_steps: int = 200000


class GoalConditionedPPOActorCritic(nn.Module):
    """
    Goal-conditioned PPO:
    입력 = [obs, goal_onehot], goal_onehot은 (2,)로 context∈{0,1}를 one-hot으로 표현
    """

    def __init__(self, obs_dim, goal_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.num_actions = num_actions

        self.fc1 = nn.Linear(obs_dim + goal_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_head = nn.Linear(hidden_dim, num_actions)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, goal_onehot):
        """
        obs: [B, obs_dim]
        goal_onehot: [B, goal_dim]
        """
        x = torch.cat([obs, goal_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi_head(x)
        value = self.v_head(x).squeeze(-1)
        return logits, value

    def act(self, obs_np, goal_idx: int, device="cpu", greedy: bool = False):
        """
        obs_np: np.array [obs_dim]
        goal_idx: 현재 rewarding goal index (0 또는 1)
        """
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(
            0
        )  # [1,obs_dim]
        goal_oh = torch.zeros(1, self.goal_dim, device=device)
        goal_oh[0, goal_idx] = 1.0

        with torch.no_grad():
            logits, value = self.forward(obs, goal_oh)
            dist = Categorical(logits=logits)
            if greedy:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())


def train_gc_ppo(env, cfg: PPOConfig, device="cpu", seed: int = 0):
    """
    GridworldAlternate에서 goal-conditioned PPO를 학습.
    policy 입력은 [obs, goal_onehot] 이고,
    goal_onehot은 env.context (rewarding goal index)를 one-hot으로 만든 것.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.obs_dim
    num_actions = env.num_actions
    goal_dim = 2  # context ∈ {0,1}

    model = GoalConditionedPPOActorCritic(obs_dim, goal_dim, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    obs, _ = env.reset()
    global_step = 0

    while global_step < cfg.total_env_steps:
        obs_buf = []
        goal_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        # rollout
        for _ in range(cfg.steps_per_rollout):
            obs_buf.append(obs.copy())
            goal_idx = env.context  # oracle처럼 현재 rewarding goal을 알고 있다고 가정
            goal_buf.append(goal_idx)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            goal_oh = torch.zeros(1, goal_dim, device=device)
            goal_oh[0, goal_idx] = 1.0

            with torch.no_grad():
                logits, value = model(obs_t, goal_oh)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            act = int(action.item())
            logp = float(log_prob.item())
            val = float(value.item())

            act_buf.append(act)
            logp_buf.append(logp)
            val_buf.append(val)

            next_obs, reward, done, info = env.step(act)

            rew_buf.append(float(reward))
            done_buf.append(float(done))

            obs = next_obs
            global_step += 1

            if done:
                obs, _ = env.reset()

            if global_step >= cfg.total_env_steps:
                break

        # GAE advantage
        with torch.no_grad():
            goal_idx = env.context
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            goal_oh = torch.zeros(1, goal_dim, device=device)
            goal_oh[0, goal_idx] = 1.0
            _, next_value = model(obs_t, goal_oh)
            next_value = float(next_value.item())

        rewards = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                v_next = next_value
            else:
                v_next = values[t + 1]
            delta = rewards[t] + cfg.gamma * (1.0 - dones[t]) * v_next - values[t]
            gae = delta + cfg.gamma * cfg.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        # Tensor 변환
        obs_tensor = torch.tensor(
            np.stack(obs_buf, axis=0), dtype=torch.float32, device=device
        )
        goal_idx_arr = np.array(goal_buf, dtype=np.int64)
        goal_tensor = torch.tensor(goal_idx_arr, dtype=torch.long, device=device)
        goal_onehot = torch.zeros(len(goal_buf), goal_dim, device=device)
        goal_onehot.scatter_(1, goal_tensor.view(-1, 1), 1.0)

        act_tensor = torch.tensor(np.array(act_buf), dtype=torch.long, device=device)
        old_logp_tensor = torch.tensor(
            np.array(logp_buf), dtype=torch.float32, device=device
        )
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        # advantage normalize
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        batch_size = obs_tensor.shape[0]
        for _ in range(cfg.ppo_epochs):
            idx = np.random.permutation(batch_size)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idx[start:end]

                mb_obs = obs_tensor[mb_idx]
                mb_goal = goal_onehot[mb_idx]
                mb_act = act_tensor[mb_idx]
                mb_old_logp = old_logp_tensor[mb_idx]
                mb_adv = adv_tensor[mb_idx]
                mb_ret = ret_tensor[mb_idx]

                logits, value = model(mb_obs, mb_goal)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_ret)

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        print(f"[GC-PPO] steps={global_step} / {cfg.total_env_steps}")
        print(rewards.mean())

    return model


# =======================================
# 3. Offline dataset (episodes + steps)
# =======================================
class Episode:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dts = []

    def add(self, obs, action, reward, dt):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dts.append(dt)


def collect_offline_dataset_ppo(
    env: GridworldAlternate, num_episodes: int = 500, seed: int = 0, device: str = "cpu"
):
    """
    1) goal-conditioned PPO policy를 학습하고
    2) 그 policy로 offline dataset을 수집.

    수집된 dataset은 여전히 (obs, a, r, obs', done)만 저장합니다.
    (offline RL 쪽에서는 goal/context 정보가 없도록)
    """
    base_dir = os.path.dirname(__file__)
    target_file = "ppo_policy.pt"
    if os.path.exists(os.path.join(base_dir, target_file)):
        random.seed(seed + 30)
        np.random.seed(seed + 30)
        torch.manual_seed(seed + 30)

        obs_dim = env.obs_dim
        num_actions = env.num_actions
        goal_dim = 2  # context ∈ {0,1}

        ppo_policy = GoalConditionedPPOActorCritic(obs_dim, goal_dim, num_actions).to(
            device
        )
        model_dict = torch.load(os.path.join(base_dir, target_file))
        ppo_policy.load_state_dict(model_dict)
    else:
        # 1) goal-conditioned PPO 학습
        ppo_cfg = PPOConfig()

        print("Training goal-conditioned PPO policy for offline data collection...")
        ppo_policy = train_gc_ppo(env, ppo_cfg, device=device, seed=seed)
        torch.save(ppo_policy.state_dict(), target_file)

    # 2) 학습된 GC-PPO policy로 dataset 수집

    episodes: List[Episode] = []
    transitions = []

    for ep in range(num_episodes):
        ep_data = Episode()
        obs, _ = env.reset()
        done = False

        while not done:
            goal_idx = env.active_goal_idx  # 현재 rewarding goal index (oracle 정보)
            action, logp, value = ppo_policy.act(
                obs, goal_idx, device=device, greedy=False
            )

            next_obs, reward, _, done, info = env.step(action)
            dt = info.get("session_changed", 0)
            dt = 1 if dt else 0

            ep_data.add(obs.copy(), action, reward, dt)

            transitions.append(
                (
                    obs.astype(np.float32),
                    int(action),
                    float(reward),
                    next_obs.astype(np.float32),
                    float(done),
                )
            )

            obs = next_obs

        ep_data.obs.append(obs.copy())
        episodes.append(ep_data)

    print(
        f"Collected {len(episodes)} episodes, {len(transitions)} transitions with goal-conditioned PPO policy."
    )
    return episodes, transitions



# =======================================
# 5. Extract latent features for each step
# =======================================


def compute_step_latents(
    episodes: List[Episode], VAE: HLT_DynaMITE_VAE, device="cpu"
):
    """
    For each step t in each episode, compute the posterior mean mu_t
    as the latent feature m_t.
    Returns:
        latent_per_step: list of np arrays aligned with transitions
                         each of shape [T, latent_dim]
    """
    VAE.eval()
    top_latent_per_episode: List[np.ndarray] = []
    latent_per_episode: List[np.ndarray] = []
    with torch.no_grad():
        for ep in episodes:
            obs = np.stack(ep.obs, axis=0)  # [T+1, obs_dim]
            acts = np.array(ep.actions, dtype=np.int64)  # [T]
            rews = np.array(ep.rewards, dtype=np.float32)  # [T]

            T = acts.shape[0]
            obs_torch = torch.tensor(
                obs[:-1], dtype=torch.float32, device=device
            ).unsqueeze(0)
            acts_torch = torch.tensor(acts, dtype=torch.long, device=device).unsqueeze(
                0
            )
            rews_torch = torch.tensor(rews, dtype=torch.float32, device=device).view(
                1, T, 1
            )

            act_prev = torch.roll(acts_torch, shifts=1, dims=1)
            act_prev[:, 0] = 0
            rew_aligned = torch.roll(rews_torch, shifts=1, dims=1)
            rew_aligned[:, 0, 0] = 0.0

            mu_top, logvar_top, mu_mid, logvar_mid, _, _ = VAE.encode(obs_torch, act_prev, rew_aligned)
            
            mu_top_np = mu_top[0].cpu().numpy()  # [T, latent_dim]
            mu_np = mu_mid[0].cpu().numpy()  # [T, latent_dim]
            
            top_latent_per_episode.append(mu_top_np)
            latent_per_episode.append(mu_np)

    return top_latent_per_episode, latent_per_episode


# =======================================
# 6. IQL networks (state + latent)
# =======================================


@dataclass
class IQLConfig:
    gamma: float = 0.99
    tau: float = 0.9  # expectile
    beta: float = 10.0  # AWR temperature
    batch_size: int = 256
    num_epochs: int = 1
    lr_q: float = 3e-4
    lr_v: float = 3e-4
    lr_pi: float = 3e-4
    adv_clip: float = 10.0


class QNet(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs, latent):
        x = torch.cat([obs, latent], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # [B, num_actions]


class VNet(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, latent):
        x = torch.cat([obs, latent], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze(-1)  # [B]


class PiNet(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(obs_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_actions)

    def forward(self, obs, latent):
        x = torch.cat([obs, latent], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # logits

    def act(self, obs, latent, greedy=True):
        with torch.no_grad():
            logits = self.forward(obs.unsqueeze(0), latent.unsqueeze(0))  # [1,A]
            probs = F.softmax(logits, dim=-1)
            if greedy:
                a = torch.argmax(probs, dim=-1)
            else:
                a = torch.distributions.Categorical(probs).sample()
        return int(a.item())


def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


def build_iql_dataset(transitions, top_latent_per_episode, latent_per_episode, device="cpu"):
    """
    transitions: list of (obs, a, r, next_obs, done) for episodes in order
    latent_per_episode: list of np arrays, each [T, latent_dim]

    We assume transitions are concatenated per episode in same order as episodes.
    """
    obs_list = []
    act_list = []
    rew_list = []
    next_obs_list = []
    done_list = []
    latent_list = []
    next_latent_list = []

    idx = 0
    for ep_latents in latent_per_episode:
        T = ep_latents.shape[0]
        for t in range(T):
            obs, a, r, next_obs, done = transitions[idx]
            obs_list.append(obs)
            act_list.append(a)
            rew_list.append(r)
            next_obs_list.append(next_obs)
            done_list.append(done)
            latent_list.append(ep_latents[t])
            # latent at next step: if t == T-1, reuse last
            if t < T - 1:
                next_latent_list.append(ep_latents[t + 1])
            else:
                next_latent_list.append(ep_latents[t])
            idx += 1

    obs_arr = torch.tensor(
        np.stack(obs_list, axis=0), dtype=torch.float32, device=device
    )
    act_arr = torch.tensor(
        np.array(act_list, dtype=np.int64), dtype=torch.long, device=device
    )
    rew_arr = torch.tensor(
        np.array(rew_list, dtype=np.float32), dtype=torch.float32, device=device
    )
    next_obs_arr = torch.tensor(
        np.stack(next_obs_list, axis=0), dtype=torch.float32, device=device
    )
    done_arr = torch.tensor(
        np.array(done_list, dtype=np.float32), dtype=torch.float32, device=device
    )
    latent_arr = torch.tensor(
        np.stack(latent_list, axis=0), dtype=torch.float32, device=device
    )
    next_latent_arr = torch.tensor(
        np.stack(next_latent_list, axis=0), dtype=torch.float32, device=device
    )

    return (
        obs_arr,
        act_arr,
        rew_arr,
        next_obs_arr,
        done_arr,
        latent_arr,
        next_latent_arr,
    )


def train_iql(
    obs_arr,
    act_arr,
    rew_arr,
    next_obs_arr,
    done_arr,
    latent_arr,
    next_latent_arr,
    num_actions,
    device="cpu",
    cfg=IQLConfig(),
    q_net=None,
    v_net=None,
    pi_net=None,
    load_path=None,
    load=False,
    iter_load:str="0"
):
    N = obs_arr.shape[0]
    dataset = torch.utils.data.TensorDataset(
        obs_arr, act_arr, rew_arr, next_obs_arr, done_arr, latent_arr, next_latent_arr
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )

    obs_dim = obs_arr.shape[1]
    latent_dim = latent_arr.shape[1]

    if q_net is None:
        q_net = QNet(obs_dim, latent_dim, num_actions).to(device)
        v_net = VNet(obs_dim, latent_dim).to(device)
        pi_net = PiNet(obs_dim, latent_dim, num_actions).to(device)
    else:
        q_net.train()
        v_net.train()
        pi_net.train()


    if load:
        load_dict={
            
            "q_net": q_net,
            "v_net": v_net,
            "pi_net": pi_net,
        }
        for load_key, load_target in load_dict.items():
            target_file = os.path.join(
                load_path, load_key + "_" + iter_load + ".pt"
            )
            if os.path.exists(target_file):
                model_dict = torch.load(target_file)
                load_target.load_state_dict(model_dict)
                print(f"{load_path}/IQL_{iter_load}.pt is loaded!")
                
    opt_q = torch.optim.Adam(q_net.parameters(), lr=cfg.lr_q)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=cfg.lr_v)
    opt_pi = torch.optim.Adam(pi_net.parameters(), lr=cfg.lr_pi)

    for epoch in range(cfg.num_epochs):
        q_losses = []
        v_losses = []
        pi_losses = []

        for batch in loader:
            s, a, r, s_next, d, z, z_next = [x.to(device) for x in batch]

            # ---- Q update ----
            q_values = q_net(s, z)  # [B,A]
            q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                v_next = v_net(s_next, z_next)
                target_q = r + cfg.gamma * (1.0 - d) * v_next

            q_loss = F.mse_loss(q_sa, target_q)

            opt_q.zero_grad()
            q_loss.backward()
            opt_q.step()

            # ---- V update ----
            with torch.no_grad():
                q_all = q_net(s, z)
                q_max = q_all.max(dim=1).values

            v_s = v_net(s, z)
            diff = q_max - v_s
            v_loss = expectile_loss(diff, cfg.tau)

            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()

            # ---- Policy update (AWR) ----
            with torch.no_grad():
                q_all = q_net(s, z)
                v_s_det = v_net(s, z)
                adv = q_all - v_s_det.unsqueeze(1)  # [B,A]
                # we only weight actions that were taken
                adv_sa = adv.gather(1, a.unsqueeze(1)).squeeze(1)
                adv_sa_clipped = torch.clamp(adv_sa, -cfg.adv_clip, cfg.adv_clip)
                weights = torch.exp(cfg.beta * adv_sa_clipped)
                weights = torch.clamp(weights, max=100.0)

            logits = pi_net(s, z)
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
            f"[IQL] Epoch {epoch+1}/{cfg.num_epochs} "
            f"Q={np.mean(q_losses):.4f} "
            f"V={np.mean(v_losses):.4f} "
            f"Pi={np.mean(pi_losses):.4f}"
        )

    return q_net, v_net, pi_net


# =======================================
# 7. Evaluation with learned belief + IQL
# =======================================


def evaluate_policy(
    env: GridworldAlternate,
    encoder: DynaMITEEncoder,
    pi_net: PiNet,
    device="cpu",
    n_episodes: int = 20,
):
    encoder.eval()
    pi_net.eval()
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0

        # RNN hidden state
        h = None
        # previous action & reward
        a_prev = 0
        r_prev = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            a_prev_t = torch.tensor([[a_prev]], dtype=torch.long, device=device)
            r_prev_t = torch.tensor([[[r_prev]]], dtype=torch.float32, device=device)

            with torch.no_grad():
                mu, logvar, term_logits, h = encoder(obs_t, a_prev_t, r_prev_t, h)
                z = mu[0, -1, :]  # [latent_dim]

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            a = pi_net.act(obs_tensor, z, greedy=True)

            next_obs, reward, _, done, info = env.step(a)

            total_r += reward

            a_prev = a
            r_prev = reward
            obs = next_obs

        returns.append(total_r)

    avg_return = sum(returns) / len(returns)
    print(f"Average return over {n_episodes} episodes: {avg_return:.2f}")
    return avg_return


# =======================================
# 8. Main entry (example usage)
# =======================================
if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    out_path = os.path.join(
        base_dir,
        "result",
        dt.datetime.now().strftime("%y%m%d"),
        dt.datetime.now().strftime("%H%M%S"),
    )
    os.makedirs(out_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = GridworldAlternate()

    # 1) goal-conditioned PPO로 offline dataset 수집
    print("Collecting offline dataset with goal-conditioned PPO policy...")
    episodes, transitions = collect_offline_dataset_ppo(
        env,
        num_episodes=200,
        seed=42,
        device=device,
    )

    T_iter = 200000
    avg_returns = []
    window = 100
    encoder, rew_dec, state_dec, q_net, v_net, pi_net = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    
    iter_start = input("Enter Iteration Start Time(default=0): ")

    if iter_start == "":
        iter_start = str(0)

    load_YMD = input("YMD: ")
    is_loaded = True
    if load_YMD != "":
        is_loaded = False
    load_HMS = input("HMS: ")

    load_path = os.path.join(base_dir,"result",load_YMD,load_HMS)

    # encoder = DynaMITEEncoder(obs_dim, num_actions, latent_dim=cfg.latent_dim).to(
    #     device
    # )
    # rew_dec = RewardDecoder(obs_dim, num_actions, cfg.latent_dim).to(device)
    # state_dec = StateDecoder(obs_dim, num_actions, cfg.latent_dim).to(device)
    
    # q_net = QNet(obs_dim, latent_dim, num_actions).to(device)
    # v_net = VNet(obs_dim, latent_dim).to(device)
    # pi_net = PiNet(obs_dim, latent_dim, num_actions).to(device)
        
    avg_returns_file = os.path.join(base_dir,"result",load_YMD,load_HMS,"avg_returns_" + iter_start + ".json")
    if os.path.exists(avg_returns_file):
        with open(avg_returns_file,"rb") as f:
            avg_returns = pickle.load(f)

    for iter in range(int(iter_start), int(iter_start) + T_iter):
        print(f"\nNow Iteration {iter+1}!")
        # 2) DynaMITE VAE 학습
        print("Training DynaMITE VAE (belief model)...")
        encoder, rew_dec, state_dec = train_dynamite_vae(
            episodes,
            obs_dim=env.obs_dim,
            num_actions=env.num_actions,
            device=device,
            encoder=encoder,
            rew_dec=rew_dec,
            state_dec=state_dec,
            load_path=load_path,
            load=False if is_loaded else True,
            iter_load=iter_start
        )

        # 3) latent 추출
        print("Computing latent features for offline dataset...")
        top_latent_per_episode, latent_per_episode = compute_step_latents(episodes, encoder, device=device)

        # 4) IQL dataset 구성
        print("Building IQL dataset...")
        (
            obs_arr,
            act_arr,
            rew_arr,
            next_obs_arr,
            done_arr,
            latent_arr,
            next_latent_arr,
        ) = build_iql_dataset(transitions, top_latent_per_episode, latent_per_episode, device=device)

        # 5) IQL 학습
        print("Training IQL on top of DynaMITE latents...")
        q_net, v_net, pi_net = train_iql(
            obs_arr,
            act_arr,
            rew_arr,
            next_obs_arr,
            done_arr,
            latent_arr,
            next_latent_arr,
            num_actions=env.num_actions,
            device=device,
            cfg=IQLConfig(),
            q_net=q_net,
            v_net=v_net,
            pi_net=pi_net,
            load_path=load_path,
            load=False if is_loaded else True,
            iter_load=iter_start
        )

        is_loaded = True
        
        # 6) 평가
        print("Evaluating learned policy with belief model + IQL policy...")
        avg_return = evaluate_policy(env, encoder, pi_net, device=device, n_episodes=20)

        if len(avg_returns) == 0:
            avg_returns.append(avg_return)
        elif len(avg_returns) < window:
            now_size = len(avg_returns)
            avg_returns.append(
                (now_size * avg_returns[-1] + avg_return) / (now_size + 1)
            )
        else:
            avg_returns.append(
                (window * avg_returns[-1] + avg_return - avg_returns[iter - window])
                / window
            )

        if (iter) % window == 0:
            plt.figure()
            plt.plot(range(len(avg_returns)), avg_returns)
            plt.title("DynaMITE-Offline(IQL), Average Reward with MA window = 100")
            plt.xlabel("Iteration")
            plt.ylabel("Average Reward")
            plt.savefig(os.path.join(out_path,"result.png"))

            save_dict = {
                "encoder": encoder,
                "rew_dec": rew_dec,
                "state_dec": state_dec,
                "q_net": q_net,
                "v_net": v_net,
                "pi_net": pi_net,
            }
            for file_name, target in save_dict.items():
                torch.save(
                    target.state_dict(),
                    os.path.join(out_path, file_name + f"_{iter}.pt"),
                )

            with open(os.path.join(out_path, f"avg_returns_{iter}.json"), "wb") as f:
                pickle.dump(avg_returns, f)
