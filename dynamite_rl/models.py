import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DynaMITE_VAE(nn.Module):
    """
    Recurrent Encoder + Reward Decoder (DynaMITE-RL style)
    - Encoder 입력: (s_t, a_{t-1}, r_t) 시퀀스
    - GRU hidden → (mu_t, logvar_t, termination_logit_t)
    - Decoder: p(r_{t+1} | s_t, a_t, m_t)
    """
    def __init__(self, obs_dim, action_dim, latent_dim, embed_dim, hidden_size):
        super().__init__()

        # --- Encoder embeddings ---
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_reward = nn.Linear(1, embed_dim)

        # Concat embeddings -> MLP -> GRU
        self.encoder_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Latent heads
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.fc_termination = nn.Linear(hidden_size, 1)  # session termination logit

        # --- Reward decoder ---
        # Input: [s_t, a_t, m_t] → r_hat
        self.reward_decoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim + latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, state, action, reward, hidden=None):
        """
        state:  (B, T, obs_dim)
        action: (B, T, action_dim)  # a_{t-1}
        reward: (B, T, 1)           # r_t
        hidden: (1, B, hidden_size) or None
        """
        e_s = F.relu(self.embed_state(state))
        e_a = F.relu(self.embed_action(action))
        e_r = F.relu(self.embed_reward(reward))

        x = torch.cat([e_s, e_a, e_r], dim=-1)
        x = self.encoder_mlp(x)

        out, hidden = self.gru(x, hidden)  # out: (B, T, hidden_size)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        term_logit = self.fc_termination(out)

        return mu, logvar, term_logit, hidden

    def decode_reward(self, state, action, latent):
        """
        state:  (B, T, obs_dim) or (N, obs_dim)
        action: (B, T, action_dim) or (N, action_dim)
        latent: (B, T, latent_dim) or (N, latent_dim)
        """
        x = torch.cat([state, action, latent], dim=-1)
        return self.reward_decoder(x)

    def reparameterize(self, mu, logvar):
        """
        mu, logvar: (..., latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class ActorCritic(nn.Module):
    """
    Policy & Value network conditioned on [state, latent m].
    - 입력: [s_t, m_t]
    - 출력: π(a|s,m), V(s,m)
    """
    def __init__(self, obs_dim, action_dim, latent_dim, hidden_size):
        super().__init__()
        input_dim = obs_dim + latent_dim

        # Actor
        self.actor = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),  # PPO에서 자주 쓰는 비선형
            init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        # Critic
        self.critic = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, 1), std=1.0),
        )

    def get_action_and_value(self, state, latent, action=None):
        """
        state:  (N, obs_dim)
        latent: (N, latent_dim)
        action: (N,)  (optional, discrete index)

        return:
          action_idx, log_prob, entropy, value
        """
        x = torch.cat([state, latent], dim=-1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)

        return action, log_prob, entropy, value
