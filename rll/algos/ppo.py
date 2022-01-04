import numpy as np
import torch
import torch.nn.functional as F
from rll.algos import rlBase
from rll.utils import nns
from rll.buffers import BufferBase
from rll.utils import mpi_tools


class rlPPO(rlBase):
    def __init__(self, actor:nns.ActorBase, critic:nns.CriticBase, optimizers, args):
        self.actor, self.critic= actor, critic
        self.optimizer_a, self.optimizer_c = optimizers
        self.train_a_itrs, self.train_c_itrs = args.train_a_itrs, args.train_c_itrs
        self.target_kl = args.target_kl

        self.proc_id = mpi_tools.proc_id()

    def action(self, x):
        return self.actor.action(x)

    def train(self, buffer:BufferBase, batch_size):

        for i in range(self.train_a_itrs):
            approx_kl = 0
            for data in buffer.get(batch_size):
                obs, act, logp_old, r2g, advantages = data['obs'], data['act'], data['logp'], data['r2g'], data['adv']
                pi = self.actor(obs)
                logp = pi.log_prob(act).sum(axis=-1)
                ratios = (logp - logp_old).exp()
                loss_a = -torch.min( advantages*ratios, advantages*torch.clamp(ratios, 1-0.2, 1+0.2) ).mean()

                approx_kl += (logp_old - logp).mean().item()

                self.optimizer_a.zero_grad()
                loss_a.backward()
                mpi_tools.mpi_avg_grads(self.actor)
                self.optimizer_a.step()
            
            approx_kl /= (buffer.buffer_size/batch_size)
            approx_kl_avg = mpi_tools.mpi_avg(approx_kl)
            if approx_kl_avg > 1.5 * self.target_kl:
                if self.proc_id == 0:
                    print('Early stopping at step %d due to reaching max kl.'%i, flush=True)
                break
        
        for _ in range(self.train_c_itrs):
            for data in buffer.get(batch_size):
                obs, r2g = data['obs'], data['r2g']
                self.optimizer_c.zero_grad()
                loss_c = F.mse_loss(self.critic(obs), r2g.view(-1,1))
                loss_c.backward()
                mpi_tools.mpi_avg_grads(self.critic)
                self.optimizer_c.step()
    