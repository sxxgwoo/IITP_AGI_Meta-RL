# agent_hlt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, kl_divergence


class DynaMITEHLTAgent:
    def __init__(self, vae, actor_critic, config):
        self.vae = vae.to(config.device)
        self.actor_critic = actor_critic.to(config.device)
        self.cfg = config

        self.opt_vae = optim.Adam(self.vae.parameters(), lr=config.lr_vae)
        self.opt_rl = optim.Adam(self.actor_critic.parameters(), lr=config.lr_policy)

    def update(self, rollouts):
        """
        rollouts: dict (B, T, ·)
          states       : (B, T, obs_dim)
          actions      : (B, T, act_dim)
          rewards      : (B, T, 1)
          prev_actions : (B, T, act_dim)
          masks        : (B, T, 1)
          log_probs    : (B, T)
          terminations : (B, T, 1)
        """
        states = rollouts["states"]
        actions = rollouts["actions"]
        rewards = rollouts["rewards"]
        prev_actions = rollouts["prev_actions"]
        masks = rollouts["masks"]
        old_log_probs_all = rollouts["log_probs"]
        terminations = rollouts["terminations"]

        B, T, _ = states.shape

        # ----------------------------------------
        # 1. Hierarchical VAE 업데이트
        # ----------------------------------------
        mu_top, logvar_top, mu_mid, logvar_mid, term_logits_mid, _ = \
            self.vae.encode(states, prev_actions, rewards)

        # reparameterization
        z_top = self.vae.reparameterize(mu_top, logvar_top)       # (B, top_dim)
        z_mid = self.vae.reparameterize(mu_mid, logvar_mid)       # (B, T, mid_dim)

        # (1) reward reconstruction
        pred_rewards = self.vae.decode_reward(states, actions, z_mid)
        loss_recon = F.mse_loss(pred_rewards, rewards, reduction="none")  # (B,T,1)
        loss_recon = (loss_recon * masks).mean()

        # (2) KL for top latent vs N(0, I)
        kl_top = -0.5 * torch.sum(
            1 + logvar_top - mu_top.pow(2) - logvar_top.exp(), dim=-1
        )  # (B,)
        loss_kl_top = kl_top.mean()

        # (3) Hierarchical KL for mid latent:
        #     q(z_mid_t | ·) vs p(z_mid_t | z_top) = N(W z_top, I)
        prior_mid_mean = self.vae.top_to_mid(z_top)  # (B, mid_dim)
        prior_mid_mean = prior_mid_mean.unsqueeze(1).expand(-1, T, -1)  # (B,T,mid_dim)

        # KL(q||p): 0.5 * sum_j[ (μ - μ0)^2 + exp(logvar) - 1 - logvar ]
        kl_mid = 0.5 * torch.sum(
            (mu_mid - prior_mid_mean).pow(2)
            + logvar_mid.exp()
            - 1.0
            - logvar_mid,
            dim=-1,   # over mid_dim
        )  # (B,T)
        loss_kl_mid = (kl_mid * masks.squeeze(-1)).mean()

        # (4) mid-level consistency (local smoothness in time)
        dist_mid_t = Normal(mu_mid[:, :-1], torch.exp(0.5 * logvar_mid[:, :-1]))
        dist_mid_t1 = Normal(mu_mid[:, 1:], torch.exp(0.5 * logvar_mid[:, 1:]))
        kl_cons_mid = kl_divergence(dist_mid_t, dist_mid_t1).sum(-1)  # (B,T-1)
        loss_consistency_mid = (
            F.relu(kl_cons_mid) * masks[:, :-1].squeeze(-1)
        ).mean()

        # (5) termination loss (mid-level)
        term_logits_mid_flat = term_logits_mid.squeeze(-1)  # (B,T)
        d_t = terminations.squeeze(-1)                      # (B,T)
        loss_term_mid = F.binary_cross_entropy_with_logits(
            term_logits_mid_flat, d_t, reduction="none"
        )
        loss_term_mid = (loss_term_mid * masks.squeeze(-1)).mean()

        # VAE total loss
        loss_vae = (
            loss_recon
            + self.cfg.kl_weight_top * loss_kl_top
            + self.cfg.kl_weight_mid * loss_kl_mid
            + self.cfg.beta_consistency_mid * loss_consistency_mid
            + self.cfg.lambda_term * loss_term_mid
        )

        self.opt_vae.zero_grad()
        loss_vae.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), self.cfg.max_grad_norm)
        self.opt_vae.step()

        # ----------------------------------------
        # 2. PPO (policy는 z_mid만 사용) 
        # ----------------------------------------
        z_mid_det = z_mid.detach()

        # --- GAE ---
        with torch.no_grad():
            flat_states_v = states.reshape(B * T, -1)
            flat_z_mid_v = z_mid_det.reshape(B * T, -1)

            _, _, _, values_flat = self.actor_critic.get_action_and_value(
                flat_states_v, flat_z_mid_v
            )
            values = values_flat.view(B, T, 1)

            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros_like(rewards[:, -1])  # (B,1)

            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = torch.zeros_like(rewards[:, t])  # (B,1)
                    nextvalues = torch.zeros_like(rewards[:, t])
                else:
                    nextnonterminal = masks[:, t + 1]
                    nextvalues = values[:, t + 1]

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

            returns = advantages + values

            # advantage normalization (valid step만)
            advantages_flat = advantages.view(-1)
            valid_mask_flat = masks.view(-1) > 0.5
            valid_adv = advantages_flat[valid_mask_flat]
            norm_adv = (valid_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
            advantages_flat[valid_mask_flat] = norm_adv
            advantages = advantages_flat.view(B, T, 1).squeeze(-1)

        # --- Flatten & valid mask ---
        flat_states = states.reshape(B * T, -1)
        flat_z_mid = z_mid_det.reshape(B * T, -1)
        flat_actions = actions.reshape(B * T, -1)
        flat_old_log_probs = old_log_probs_all.reshape(B * T)
        flat_advantages = advantages.reshape(B * T)
        flat_returns = returns.reshape(B * T)
        flat_masks = masks.reshape(B * T)

        valid = flat_masks > 0.5
        flat_states = flat_states[valid]
        flat_z_mid = flat_z_mid[valid]
        flat_actions = flat_actions[valid]
        flat_old_log_probs = flat_old_log_probs[valid]
        flat_advantages = flat_advantages[valid]
        flat_returns = flat_returns[valid]

        total_samples = flat_states.shape[0]
        idxs = np.arange(total_samples)

        # --- PPO epochs ---
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, total_samples, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = idxs[start:end]

                mb_states = flat_states[mb_idx]
                mb_z_mid = flat_z_mid[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_old_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                action_idx = mb_actions.argmax(-1)

                _, new_log_probs, entropy, new_values = \
                    self.actor_critic.get_action_and_value(
                        mb_states, mb_z_mid, action=action_idx
                    )
                new_values = new_values.squeeze(-1)

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
