# import torch

# class Config:
#     # --- General ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     seed = 42
    
#     # --- Environment (GridWorld) ---
#     env_name = "GridWorldAlternate"
#     max_episode_length = 60
    
#     # --- DynaMITE-RL Specific ---
#     bernoulli_p = 0.07 
    
#     latent_dim = 5
#     embedding_dim = 8
    
#     # --- PPO & Optimization ---
#     num_iterations = 1000
#     num_processes = 16
    
#     ppo_epochs = 10
#     mini_batch_size = 256
    
#     lr_policy = 3e-4
#     lr_vae = 3e-4
    
#     gamma = 0.99
#     gae_lambda = 0.95
#     max_grad_norm = 0.5
#     ppo_clip_eps = 0.2
    
#     # [Tuning 1] 탐험을 더 많이 하도록 상향 (기존 0.01)
#     entropy_coef = 0.05  
    
#     value_loss_coef = 0.5
    
#     # --- Loss Weights ---
#     # [Tuning 2] 변화에 더 빨리 적응하도록 하향 (기존 0.5)
#     # 현재 코드엔 세션 감지 로직이 없으므로, 이 값이 너무 크면 방해가 됩니다.
#     beta_consistency = 0.05  
    
#     lambda_vae = 0.01
    
#     # [Tuning 3] 모델 용량 증가 (기존 64)
#     vae_hidden_size = 128
#     actor_critic_hidden = 128



import torch

class Config:
    # --- General ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    
    # --- Environment (GridWorld) ---
    env_name = "GridWorldAlternate"
    max_episode_length = 60
    
    # --- DynaMITE-RL Specific ---
    bernoulli_p = 0.07 
    
    latent_dim = 5
    embedding_dim = 8
    
    # --- PPO & Optimization ---
    # [핵심] 28점이 나오게 만든 설정 (PPO 반복 학습)
    num_iterations = 1000   
    num_processes = 16
    
    ppo_epochs = 10        # 데이터 재사용 횟수
    mini_batch_size = 256  # 미니배치 크기
    
    lr_policy = 3e-4
    lr_vae = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    max_grad_norm = 0.5
    ppo_clip_eps = 0.2
    entropy_coef = 0.01    # (튜닝 전: 0.01)
    value_loss_coef = 0.5
    
    # --- Loss Weights ---
    beta_consistency = 0.5 # (튜닝 전: 0.5)
    lambda_vae = 0.01
    lambda_term = 1.0           # NEW: termination BCE loss weight
    
    # Hidden sizes
    vae_hidden_size = 64   # (튜닝 전: 64)
    actor_critic_hidden = 128