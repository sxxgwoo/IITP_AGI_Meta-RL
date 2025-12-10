import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence


class DynaMITEAgent:
    def __init__(self, vae, actor_critic, config):
        self.vae = vae.to(config.device)
        self.actor_critic = actor_critic.to(config.device)
        self.cfg = config

        self.opt_vae = optim.Adam(self.vae.parameters(), lr=config.lr_vae)
        self.opt_rl = optim.Adam(self.actor_critic.parameters(), lr=config.lr_policy)

    def update(self, rollouts):
        """
        rollouts: dict containing tensors of shape (B, T, ·)
        keys:
          states       : (B, T, obs_dim)
          actions      : (B, T, act_dim)  (one-hot)
          rewards      : (B, T, 1)
          prev_actions : (B, T, act_dim)
          masks        : (B, T, 1)   (1 for valid, 0 for padding/terminal 이후)
          log_probs    : (B, T)
          terminations : (B, T, 1)   (session_changed d_t)
        """
        states = rollouts["states"]         # (B, T, obs_dim)
        actions = rollouts["actions"]       # (B, T, act_dim)
        rewards = rollouts["rewards"]       # (B, T, 1)
        prev_actions = rollouts["prev_actions"]  # (B, T, act_dim)
        masks = rollouts["masks"]           # (B, T, 1)
        old_log_probs_all = rollouts["log_probs"]  # (B, T)
        terminations = rollouts["terminations"]    # (B, T, 1)

        B, T, _ = states.shape

        # ---------------------------------------------------
        # 1. Train VAE (Inference Model) - One Pass per Iter
        # ---------------------------------------------------
        mu, logvar, term_logits, _ = self.vae.encode(states, prev_actions, rewards)
        z = self.vae.reparameterize(mu, logvar)

        # A. Reconstruction Loss (reward reconstruction)
        pred_rewards = self.vae.decode_reward(states, actions, z)
        loss_recon = F.mse_loss(pred_rewards, rewards, reduction="none")  # (B, T, 1)
        loss_recon = (loss_recon * masks).mean()

        # B. KL Divergence w.r.t N(0, I)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # (B, T)
        loss_kl = (loss_kl * masks.squeeze(-1)).mean()

        # C. Consistency Loss (local consecutive KL)
        dist_t = Normal(mu[:, :-1], torch.exp(0.5 * logvar[:, :-1]))
        dist_t1 = Normal(mu[:, 1:], torch.exp(0.5 * logvar[:, 1:]))
        kl_consecutive = kl_divergence(dist_t, dist_t1).sum(-1)  # (B, T-1)
        loss_consistency = (
            F.relu(kl_consecutive) * masks[:, :-1].squeeze(-1)
        ).mean()

        # D. Termination Loss (binary cross-entropy on d_t)
        term_logits_flat = term_logits.squeeze(-1)      # (B, T)
        d_t = terminations.squeeze(-1)                 # (B, T)
        loss_term = F.binary_cross_entropy_with_logits(
            term_logits_flat, d_t, reduction="none"
        )
        loss_term = (loss_term * masks.squeeze(-1)).mean()

        # Final VAE Loss
        loss_vae = (
            loss_recon
            + self.cfg.beta_consistency * loss_consistency
            + 0.1 * loss_kl
            + self.cfg.lambda_term * loss_term
        )

        self.opt_vae.zero_grad()
        loss_vae.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), self.cfg.max_grad_norm)
        self.opt_vae.step()

        # ---------------------------------------------------
        # 2. Train Policy (PPO) - Multiple Epochs
        # ---------------------------------------------------
        # Detach z for RL (Treat latent as fixed state feature for this update)
        z_detached = z.detach()

        # --- GAE 계산 ---
        with torch.no_grad():
            # ActorCritic는 (N, obs_dim)를 기대하므로 flatten해서 value 계산
            flat_states_v = states.reshape(B * T, -1)
            flat_z_v = z_detached.reshape(B * T, -1)
            _, _, _, values_flat = self.actor_critic.get_action_and_value(
                flat_states_v, flat_z_v
            )
            values = values_flat.view(B, T, 1)  # (B, T, 1)

            advantages = torch.zeros_like(rewards)  # (B, T, 1)
            lastgaelam = torch.zeros_like(rewards[:, -1])  # (B, 1)

            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = torch.zeros_like(rewards[:, t])  # (B,1)
                    nextvalues = torch.zeros_like(rewards[:, t])       # (B,1)
                else:
                    nextnonterminal = masks[:, t + 1]      # (B,1)
                    nextvalues = values[:, t + 1]          # (B,1)

                delta = (
                    rewards[:, t]
                    + self.cfg.gamma * nextvalues * nextnonterminal
                    - values[:, t]
                )
                lastgaelam = (
                    delta
                    + self.cfg.gamma
                    * self.cfg.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
                advantages[:, t] = lastgaelam

            returns = advantages + values  # (B, T, 1)

            # Advantage Normalization
            advantages_flat = advantages.view(-1)
            # 마스크된 부분만 사용해서 평균/표준편차 계산
            valid_mask_flat = masks.view(-1) > 0.5
            valid_adv = advantages_flat[valid_mask_flat]
            norm_adv = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
            advantages_flat[valid_mask_flat] = norm_adv
            advantages = advantages_flat.view(B, T, 1)
            advantages = advantages.squeeze(-1)  # (B, T)

        # --- Flatten data for mini-batch PPO ---
        flat_states = states.reshape(B * T, -1)
        flat_z = z_detached.reshape(B * T, -1)
        flat_actions = actions.reshape(B * T, -1)
        flat_log_probs = old_log_probs_all.reshape(B * T)
        flat_advantages = advantages.reshape(B * T)
        flat_returns = returns.reshape(B * T)
        flat_masks = masks.reshape(B * T)  # (B*T,1) → (B*T,)

        # padding / invalid timestep 제거
        valid = flat_masks > 0.5
        flat_states = flat_states[valid]
        flat_z = flat_z[valid]
        flat_actions = flat_actions[valid]
        flat_log_probs = flat_log_probs[valid]
        flat_advantages = flat_advantages[valid]
        flat_returns = flat_returns[valid]

        total_samples = flat_states.shape[0]
        idxs = np.arange(total_samples)

        # --- PPO Epochs Loop ---
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, total_samples, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = idxs[start:end]

                mb_states = flat_states[mb_idx]
                mb_z = flat_z[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                # 현재 policy 평가
                action_idx = mb_actions.argmax(-1)  # (mb,)
                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                    mb_states, mb_z, action=action_idx
                )
                new_values = new_values.squeeze(-1)  # (mb,)

                # PPO ratio & surrogate
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.ppo_clip_eps,
                    1.0 + self.cfg.ppo_clip_eps,
                ) * mb_advantages

                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = self.cfg.value_loss_coef * F.mse_loss(
                    new_values, mb_returns
                )
                loss_entropy = -self.cfg.entropy_coef * entropy.mean()

                loss_rl = loss_actor + loss_critic + loss_entropy

                self.opt_rl.zero_grad()
                loss_rl.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.opt_rl.step()

        return {
            "loss_vae": loss_vae.item(),
            "loss_rl": loss_rl.item(),
            "reward": rewards.sum().item(),
        }
