import torch.nn as nn
import torch
from torch.distributions import Categorical
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
