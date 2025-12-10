# main_hlt.py
import os
from datetime import datetime

import numpy as np
import torch

from config import ConfigHLT as Config
from envs import GridWorldAlternate
from models import HLT_DynaMITE_VAE, HLT_ActorCritic
from agent import DynaMITEHLTAgent

from torch.utils.tensorboard import SummaryWriter

'''
python main.py
# 또는
nohup python -u main.py > hlt_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

'''
def make_env():
    return GridWorldAlternate(
        max_steps=Config.max_episode_length,
        switch_prob=Config.bernoulli_p,
    )


def evaluate(env, agent, config, num_episodes=5):
    agent.vae.eval()
    agent.actor_critic.eval()

    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        h = None
        prev_action = torch.zeros(env.action_space.n, device=config.device)
        prev_reward = torch.zeros(1, device=config.device)
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=config.device
        ).unsqueeze(0)

        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                # HLT 인코딩
                mu_top, logvar_top, mu_mid, logvar_mid, _, h = agent.vae.encode(
                    obs_tensor.unsqueeze(1),           # (1, 1, obs_dim)
                    prev_action.view(1, 1, -1),        # (1, 1, act_dim)
                    prev_reward.view(1, 1, -1),        # (1, 1, 1)
                    hidden=h
                )
                # mid-level latent만 사용 (policy input)
                z_mid_all = agent.vae.reparameterize(mu_mid, logvar_mid)  # (1, 1, latent_mid_dim)
                z_mid = z_mid_all.squeeze(1)                              # (1, latent_mid_dim)

                action_idx, _, _, _ = agent.actor_critic.get_action_and_value(
                    obs_tensor,  # (1, obs_dim)
                    z_mid        # (1, latent_mid_dim) → obs+latent = 6+5=11 OK
                )

                action = action_idx.item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            action_one_hot = torch.zeros(
                env.action_space.n, device=config.device
            )
            action_one_hot[action] = 1.0

            obs_tensor = torch.as_tensor(
                next_obs, dtype=torch.float32, device=config.device
            ).unsqueeze(0)
            prev_action = action_one_hot
            prev_reward = torch.as_tensor(
                [reward], dtype=torch.float32, device=config.device
            )

            ep_reward += reward

        total_rewards.append(ep_reward)

    agent.vae.train()
    agent.actor_critic.train()

    return float(np.mean(total_rewards)), float(np.std(total_rewards))


def collect_rollouts(env, agent, num_episodes, config):
    rollout_buffer = {
        "states": [],
        "actions": [],
        "rewards": [],
        "prev_actions": [],
        "masks": [],
        "log_probs": [],
        "terminations": [],
    }

    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_prev_actions = []
        ep_masks = []
        ep_log_probs = []
        ep_terminations = []

        h = None
        prev_action = torch.zeros(env.action_space.n, device=config.device)
        prev_reward = torch.zeros(1, device=config.device)
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=config.device
        ).unsqueeze(0)

        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                mu_top, logvar_top, mu_mid, logvar_mid, _, h = agent.vae.encode(
                    obs_tensor.unsqueeze(1),
                    prev_action.view(1, 1, -1),
                    prev_reward.view(1, 1, -1),
                    hidden=h
                )
                z_mid_all = agent.vae.reparameterize(mu_mid, logvar_mid)  # (1, 1, latent_mid_dim)
                z_mid = z_mid_all.squeeze(1)                              # (1, latent_mid_dim)

                action_idx, log_prob, _, _ = agent.actor_critic.get_action_and_value(
                    obs_tensor,
                    z_mid
                )

                action = action_idx.item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            session_changed = info.get("session_changed", False)

            action_one_hot = torch.zeros(
                env.action_space.n, device=config.device
            )
            action_one_hot[action] = 1.0

            ep_states.append(obs_tensor)
            ep_actions.append(action_one_hot.unsqueeze(0))
            ep_rewards.append(
                torch.tensor([[reward]], dtype=torch.float32, device=config.device)
            )
            ep_prev_actions.append(prev_action.unsqueeze(0))
            ep_masks.append(
                torch.tensor(
                    [[1.0 if not done else 0.0]],
                    dtype=torch.float32,
                    device=config.device,
                )
            )
            ep_log_probs.append(log_prob)
            ep_terminations.append(
                torch.tensor(
                    [[1.0 if session_changed else 0.0]],
                    dtype=torch.float32,
                    device=config.device,
                )
            )

            obs_tensor = torch.as_tensor(
                next_obs, dtype=torch.float32, device=config.device
            ).unsqueeze(0)
            prev_action = action_one_hot
            prev_reward = torch.as_tensor(
                [reward], dtype=torch.float32, device=config.device
            )

            ep_reward += reward

        total_rewards.append(ep_reward)

        rollout_buffer["states"].append(torch.cat(ep_states, dim=0))
        rollout_buffer["actions"].append(torch.cat(ep_actions, dim=0))
        rollout_buffer["rewards"].append(torch.cat(ep_rewards, dim=0))
        rollout_buffer["prev_actions"].append(torch.cat(ep_prev_actions, dim=0))
        rollout_buffer["masks"].append(torch.cat(ep_masks, dim=0))
        rollout_buffer["log_probs"].append(torch.cat(ep_log_probs, dim=0))
        rollout_buffer["terminations"].append(torch.cat(ep_terminations, dim=0))

    from torch.nn.utils.rnn import pad_sequence

    batch_data = {}
    for k, v in rollout_buffer.items():
        batch_data[k] = pad_sequence(v, batch_first=True).to(config.device)

    return batch_data, float(np.mean(total_rewards))


def main():
    cfg = Config()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("checkpoints_hlt", f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    base_log_dir = os.path.join("logs", "dynamite_gridworld_hlt")
    log_dir_run = os.path.join(base_log_dir, f"run_{run_id}")
    os.makedirs(log_dir_run, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_run)

    print(f"[HLT] Initializing DynaMITE-RL (Hierarchical) for {cfg.env_name}...")
    print(f"Checkpoints: {save_dir}")
    print(f"TensorBoard: {log_dir_run}")

    env = make_env()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    vae = HLT_DynaMITE_VAE(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_top_dim=cfg.latent_top_dim,
        latent_mid_dim=cfg.latent_mid_dim,
        embed_dim=cfg.embedding_dim,
        hidden_size=cfg.vae_hidden_size,
    )

    actor_critic = HLT_ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        # latent_top_dim=cfg.latent_top_dim,
        latent_mid_dim=cfg.latent_mid_dim,
        hidden_size=cfg.actor_critic_hidden,
    )

    agent = DynaMITEHLTAgent(vae, actor_critic, cfg)

    best_eval_reward = -float("inf")
    print("[HLT] Starting Training...")

    for i in range(1, cfg.num_iterations + 1):
        batch_data, train_avg_reward = collect_rollouts(
            env, agent, num_episodes=cfg.num_processes, config=cfg
        )

        metrics = agent.update(batch_data)

        writer.add_scalar("Train/avg_episode_reward", train_avg_reward, i)
        writer.add_scalar("Loss/vae", metrics["loss_vae"], i)
        writer.add_scalar("Loss/rl", metrics["loss_rl"], i)

        if i % 10 == 0:
            eval_reward, eval_std = evaluate(env, agent, cfg, num_episodes=10)

            print(
                f"[HLT] Iter {i:4d} | "
                f"Train R: {train_avg_reward:.2f} | "
                f"Eval R: {eval_reward:.2f} (+/- {eval_std:.2f}) | "
                f"VAE Loss: {metrics['loss_vae']:.4f} | "
                f"RL Loss: {metrics['loss_rl']:.4f}"
            )

            writer.add_scalar("Eval/avg_episode_reward", eval_reward, i)
            writer.add_scalar("Eval/episode_reward_std", eval_std, i)
            writer.add_scalar(
                "Eval/best_avg_episode_reward",
                max(best_eval_reward, eval_reward),
                i,
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_path = os.path.join(
                    save_dir, f"best_model_{run_id}_iter{i}.pt"
                )
                torch.save(
                    {
                        "iter": i,
                        "vae_state_dict": agent.vae.state_dict(),
                        "actor_critic_state_dict": agent.actor_critic.state_dict(),
                        "best_reward": best_eval_reward,
                    },
                    save_path,
                )
                print(f"    >>> [HLT] New Best Saved: {save_path}! Reward: {best_eval_reward:.2f}")

    print(f"[HLT] Training Complete. Best Eval Reward: {best_eval_reward:.2f}")
    writer.close()


if __name__ == "__main__":
    main()
