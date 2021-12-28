import numpy as np
import torch
import torch.nn.functional as F
from rll.algos import rlBase
from rll.utils import nns
from rll.buffers import BufferBase
from rll.utils import mpi_tools


class rlPPO(rlBase):
    def __init__(self, actor:nns.ActorBase, critic:nns.CriticBase, optimizers, train_itrs):
        self.actor, self.critic= actor, critic
        self.optimizer_a, self.optimizer_c = optimizers
        self.train_a_itrs, self.train_c_itrs = train_itrs

    def action(self, x):
        return self.actor.action(x)

    def train(self, buffer:BufferBase):
        data = buffer.get()
        obs, act, logp_old, r2g, advantages = data['obs'], data['act'], data['logp'], data['r2g'], data['adv']

        # baselines = self.critic(obs).detach().view(-1)
        # advantages = r2g - baselines

        for _ in range(self.train_a_itrs):
            pi = self.actor(obs)
            logp = pi.log_prob(act).sum(axis=-1)
            ratios = (logp - logp_old).exp()
            loss_a = -torch.min( advantages*ratios, advantages*torch.clamp(ratios, 1-0.2, 1+0.2) ).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            mpi_tools.mpi_avg_grads(self.actor)
            self.optimizer_a.step()
        
        for _ in range(self.train_c_itrs):
            self.optimizer_c.zero_grad()
            loss_c = F.mse_loss(self.critic(obs), r2g.view(-1,1))
            loss_c.backward()
            mpi_tools.mpi_avg_grads(self.critic)
            self.optimizer_c.step()
    