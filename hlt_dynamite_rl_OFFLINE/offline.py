import os
import random
import numpy as np
import torch

from envs import GridWorldAlternate
from episode import Episode
from GoalPPO import GoalConditionedPPOActorCritic
from config import PPOConfig

from typing import List

from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn


def train_gc_ppo(env, cfg: PPOConfig, device="cpu", seed: int = 0):
    """
    GridworldAlternate에서 goal-conditioned PPO를 학습.
    policy 입력은 [obs, goal_onehot] 이고,
    goal_onehot은 env.context (rewarding goal index)를 one-hot으로 만든 것.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = 6
    num_actions = 4
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
            goal_idx = env.active_goal_idx  # oracle처럼 현재 rewarding goal을 알고 있다고 가정
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
            goal_idx = env.active_goal_idx
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

        obs_dim = 6
        num_actions = 4
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

            next_obs, reward, done, info = env.step(action)
            dt = info.get("dt", 0)

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