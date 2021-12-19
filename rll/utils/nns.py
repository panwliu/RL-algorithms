import torch
import numpy as np
import torch.nn.functional as F

def mlp(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i+1]), act()]
    return torch.nn.Sequential(*layers)

class ActorBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def action(self, x):
        raise NotImplementedError

class CriticBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError


class MLPCategoricalActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, x):
        pi = torch.distributions.Categorical(logits=self.logits_net(x))
        return pi

class MLPGaussianActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, std):
        super().__init__()
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.std = torch.tensor(std, dtype=torch.float32)

    def forward(self, x):
        pi = torch.distributions.Normal(loc=self.mu_net(x), scale=self.std)
        return pi
    
    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pi = self.forward(x)
        a = pi.sample().detach()
        logp = pi.log_prob(a).detach().sum(axis=-1)
        return a.numpy(), logp.numpy()

class MLPCritic(torch.nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, x):
        return self.v_net(x)

class MLPQFunction(torch.nn.Module):                # for discrete acitons
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.obs_dim = obs_dim
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, x):
        return self.q_net(x)
    
    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.obs_dim)
        q_values = self.forward(x).detach()
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action.numpy()

class MLPQFunction2(torch.nn.Module):               # for continous acitons
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim+act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        return self.q_net( torch.cat([obs, act], dim=-1) ).squeeze(dim=-1)

class MLPDeterministicActor(torch.nn.Module):      # for continous acitons
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, x):
        return self.mu_net(x)

class MLPActorCritic2(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        self.q1 = MLPQFunction2(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction2(obs_dim, act_dim, hidden_sizes, activation)
        self.pi = MLPDeterministicActor(obs_dim, act_dim, hidden_sizes, activation)

    def action(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

LOG_STD_MAX = 2
LOG_STD_MIN = -20
class SquashedGaussianMLPActor(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = torch.nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = torch.nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = torch.distributions.Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPActorCritic3(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit=1.0)
        self.q1 = MLPQFunction2(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction2(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()