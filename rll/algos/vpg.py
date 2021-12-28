import numpy as np
import torch
import torch.nn.functional as F
from rll.algos import rlBase
from rll.utils import nns
from rll.buffers import BufferBase
from rll.utils import mpi_tools

class rlVPG(rlBase):
    def __init__(self, actor:nns.ActorBase, critic:nns.CriticBase, optimizers, train_itrs) -> None:
        super().__init__()
        self.actor, self.critic = actor, critic
        self.optim_a, self.optim_c = optimizers
        self.train_a_itrs, self.train_c_itrs = train_itrs
    
    def action(self, x:np.ndarray):
        return self.actor.action(x)

    def train(self, buffer:BufferBase):
        data = buffer.get()
        obs, act, r2g, advantages = data['obs'], data['act'], data['r2g'], data['adv']

        for _ in range(self.train_a_itrs):
            pi = self.actor(obs)
            logp = pi.log_prob(act).sum(axis=-1)
            loss = -(logp * advantages).mean()

            self.optim_a.zero_grad()
            loss.backward()
            mpi_tools.mpi_avg_grads(self.actor)
            self.optim_a.step()
        
        for _ in range(self.train_c_itrs):
            self.optim_c.zero_grad()
            loss_c = F.mse_loss(self.critic(obs), r2g.view(-1,1))
            loss_c.backward()
            mpi_tools.mpi_avg_grads(self.critic)
            self.optim_c.step()


