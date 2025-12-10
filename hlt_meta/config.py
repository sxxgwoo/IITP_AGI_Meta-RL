# config_hlt.py
import torch


class ConfigHLT:
    # --- General ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    # --- Environment (GridWorld) ---
    env_name = "GridWorldAlternate"
    max_episode_length = 60

    # --- DynaMITE Gridworld ---
    bernoulli_p = 0.07  # latent switch prob (session change)

    # Hierarchical latent sizes
    latent_top_dim = 2   # 상위 latent, 작게
    latent_mid_dim = 5   # 기존 m_t 역할
    embedding_dim = 8

    # --- PPO & Optimization ---
    num_iterations = 1000
    num_processes = 16

    ppo_epochs = 10
    mini_batch_size = 256

    lr_policy = 3e-4
    lr_vae = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    max_grad_norm = 0.5
    ppo_clip_eps = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.5

    # --- VAE Loss Weights ---
    kl_weight_top = 0.01     # 상위 KL은 아주 약하게
    kl_weight_mid = 0.1      # 중간 KL (hierarchical prior)
    beta_consistency_mid = 0.5
    lambda_term = 1.0        # termination BCE

    # --- Hidden sizes ---
    vae_hidden_size = 64
    actor_critic_hidden = 128
