# models_hlt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HLT_DynaMITE_VAE(nn.Module):
    """
    Hierarchical Latent VAE (stable version):
      - top latent z_top: sequence-level context
      - mid latent z_mid_t: step-level belief (policy/decoder에서 사용)
      - hierarchy: z_mid_t의 prior mean = W * z_top  (learned mapping)
    """
    def __init__(self, obs_dim, action_dim,
                 latent_top_dim, latent_mid_dim,
                 embed_dim, hidden_size):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_top_dim = latent_top_dim
        self.latent_mid_dim = latent_mid_dim
        self.hidden_size = hidden_size

        # --- Encoder embeddings ---
        self.embed_state = nn.Linear(obs_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_reward = nn.Linear(1, embed_dim)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # --- Top-level latent heads (sequence-level) ---
        self.fc_mu_top = nn.Linear(hidden_size, latent_top_dim)
        self.fc_logvar_top = nn.Linear(hidden_size, latent_top_dim)

        # --- Mid-level latent heads (step-level) ---
        self.fc_mu_mid = nn.Linear(hidden_size, latent_mid_dim)
        self.fc_logvar_mid = nn.Linear(hidden_size, latent_mid_dim)

        # Mid-level termination (session change)
        self.fc_term_mid = nn.Linear(hidden_size, 1)

        # --- Hierarchical prior: top → mid ---
        # top latent를 mid latent의 prior mean으로 매핑
        self.top_to_mid = nn.Linear(latent_top_dim, latent_mid_dim)

        # --- Reward decoder ---
        # policy/decoder는 mid latent만 사용 (RL 안정성 위해)
        dec_in_dim = obs_dim + action_dim + latent_mid_dim
        self.reward_decoder = nn.Sequential(
            nn.Linear(dec_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode(self, state, action, reward, hidden=None):
        """
        state:  (B, T, obs_dim)
        action: (B, T, action_dim)  # a_{t-1}
        reward: (B, T, 1)           # r_t

        returns:
          mu_top, logvar_top : (B, latent_top_dim)
          mu_mid, logvar_mid : (B, T, latent_mid_dim)
          term_logits_mid    : (B, T, 1)
          hidden             : (1, B, H)
        """
        e_s = F.relu(self.embed_state(state))
        e_a = F.relu(self.embed_action(action))
        e_r = F.relu(self.embed_reward(reward))

        x = torch.cat([e_s, e_a, e_r], dim=-1)
        x = self.encoder_mlp(x)

        out, hidden = self.gru(x, hidden)  # (B, T, H)

        h_last = out[:, -1, :]  # sequence summary

        mu_top = self.fc_mu_top(h_last)
        logvar_top = self.fc_logvar_top(h_last)

        mu_mid = self.fc_mu_mid(out)
        logvar_mid = self.fc_logvar_mid(out)

        term_logits_mid = self.fc_term_mid(out)

        return mu_top, logvar_top, mu_mid, logvar_mid, term_logits_mid, hidden

    def decode_reward(self, state, action, z_mid):
        """
        state: (B, T, obs_dim)
        action: (B, T, action_dim)
        z_mid: (B, T, latent_mid_dim)
        """
        x = torch.cat([state, action, z_mid], dim=-1)
        return self.reward_decoder(x)

    @staticmethod
    def reparameterize(mu, logvar):
        """
        mu, logvar: (..., latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class HLT_ActorCritic(nn.Module):
    """
    Policy & Value network conditioned on [state, z_mid].
    (z_top은 mid의 prior에만 쓰이고, policy에는 직접 안 들어감)
    """
    def __init__(self, obs_dim, action_dim, latent_mid_dim, hidden_size):
        super().__init__()
        input_dim = obs_dim + latent_mid_dim

        self.actor = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, action_dim), std=0.01),
        )

        self.critic = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_layer(nn.Linear(hidden_size, 1), std=1.0),
        )

        # action space는 main에서 env.action_space.n으로 맞춰줌

    def get_action_and_value(self, state, z_mid, action=None):
        """
        state : (N, obs_dim)
        z_mid : (N, latent_mid_dim)
        """
        x = torch.cat([state, z_mid], dim=-1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(x)

        return action, log_prob, entropy, value
