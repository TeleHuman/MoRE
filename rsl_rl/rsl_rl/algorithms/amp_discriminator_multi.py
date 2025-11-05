import torch
import torch.nn as nn
from torch import autograd
import torch.nn.utils.spectral_norm as spectral_norm


class AMPDiscriminatorMulti(nn.Module):
    def __init__(
            self, state_dim, amp_reward_coef, hidden_layer_sizes, device, 
            num_frames=2, task_reward_lerp=0.0, use_lerp=True):
        super(AMPDiscriminatorMulti, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.use_lerp = use_lerp
        self.num_frames = num_frames    # length of state sequence for computing amp reward

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        
        curr_in_dim = state_dim * num_frames
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(spectral_norm(nn.Linear(curr_in_dim, hidden_dim)))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = spectral_norm(nn.Linear(hidden_layer_sizes[-1], 1)).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_states,
                         lambda_=10):
        """
        expert_states: torch.Tensor, shape=(num_envs, num_frames, num_amp_obs)
        """
        expert_data = expert_states.flatten(1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def get_disc_weights(self):
        weights = []
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self.amp_linear.weight))
        return weights

    def get_disc_logit_weights(self):
        return torch.flatten(self.amp_linear.weight)

    def predict_amp_reward(
            self, states,
            task_reward, normalizer=None):
        """
            states: torch.Tensor, shape=(num_envs, num_frames, num_amp_obs)
            task_reward: torch.Tensor, shape=(num_envs, 1)
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                states = normalizer.normalize_torch(states, self.device)
            
            state_cat = states.flatten(1)
            d = self.amp_linear(self.trunk(state_cat))
            disc_reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            
            if self.use_lerp:
                if self.task_reward_lerp > 0:
                    reward = self._lerp_reward(disc_reward, task_reward.unsqueeze(-1))
                self.train()
                return reward.squeeze(), d, disc_reward.squeeze() * (1.0 - self.task_reward_lerp)
            else:
                disc_reward *= 0.02
                reward = task_reward.unsqueeze(-1) + disc_reward
                self.train()
                return reward.squeeze(), d, disc_reward.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r