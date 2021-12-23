import numpy as np
import torch
from rll.algos import rlBase
from rll.utils import nns
from rll.buffers import BufferBase
from rll.utils import mpi_tools

class rlReinforce(rlBase):
    def __init__(self, actor:nns.ActorBase, optimizer:torch.optim.Optimizer) -> None:
        super().__init__()

        self.actor, self.critic = actor, None
        self.optim = optimizer
    
    def action(self, x:np.ndarray):
        return self.actor.action(x)

    def train(self, buffer:BufferBase):
        obs, acts, rewards, _, r2g, _ = buffer.get()

        for _ in range(1):
            pi = self.actor(obs)
            logp = pi.log_prob(acts).sum(axis=-1)
            # loss = -(logp * r2g).mean()
            loss = -(logp * (r2g-r2g.mean())).mean()

            self.optim.zero_grad()
            loss.backward()
            mpi_tools.mpi_avg_grads(self.actor)
            self.optim.step()

    def eval(self):
        pass

