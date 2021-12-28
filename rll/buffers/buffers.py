import numpy as np
import torch
from rll.utils import mpi_tools

class BufferBase:
    def store(self):
        raise NotImplementedError
    def get(self):
        raise NotImplementedError

    def discount_cumsum(self, x, discount, last_val=0):
        x_cum = np.zeros_like(x)

        x_cum[-1] = x[-1] + last_val
        for k in reversed(range(len(x)-1)):
            x_cum[k] = x[k] + discount*x_cum[k+1]
        
        return x_cum

# for policy gradient based algorithms, e.g. reinforce, vpg, ppo
class ExpBuffer(BufferBase):
    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((buffer_size,obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size,act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.obs_next_buf = np.zeros((buffer_size,obs_dim), dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)

        self.reward_to_go_buf = np.zeros(buffer_size, dtype=np.float32)
        self.advantage_buf = np.zeros(buffer_size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.traj_start_idx = 0, 0

    def store(self, obs, act, reward, obs_next, logp):
        self.obs_buf[self.ptr,:] = obs
        self.act_buf[self.ptr,:] = act
        self.reward_buf[self.ptr] = reward
        self.obs_next_buf[self.ptr,:] = obs_next
        self.logp_buf[self.ptr] = logp
        
        self.ptr += 1
    
    def finish_traj(self, critic=None, last_val=0):
        obs = self.obs_buf[self.traj_start_idx:self.ptr]
        rewards = self.reward_buf[self.traj_start_idx:self.ptr]

        r_cum = self.discount_cumsum(rewards, self.gamma, last_val)
        self.reward_to_go_buf[self.traj_start_idx:self.ptr] = r_cum

        if critic:
            vals = self.critic(torch.as_tensor(obs,dtype=torch.float32)).detach().numpy()
            vals = np.append(vals, last_val)
            deltas = rewards + self.gamma * vals[1:] - vals[:-1]

            advantages = self.discount_cumsum(deltas, self.gamma*self.lam)
            self.advantage_buf[self.traj_start_idx:self.ptr] = advantages

        self.traj_start_idx = self.ptr

    def get(self):
        self.ptr, self.traj_start_idx = 0, 0

        adv_mean, adv_std = mpi_tools.mpi_statistics_scalar(self.advantage_buf)
        self.advantage_buf = (self.advantage_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, reward=self.reward_buf, obs2=self.obs_next_buf,
                    logp=self.logp_buf, r2g=self.reward_to_go_buf, adv=self.advantage_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


# for value function based RL algorithms, e.g. DQN
class ReplayBuffer(BufferBase):
    def __init__(self, obs_dim, act_dim, buffer_size):
        self.buffer_size = buffer_size
        self.obs_buf = np.zeros((buffer_size,obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size,act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.obs_next_buf = np.zeros((buffer_size,obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr, self.buffer_filled = 0, False

    def store(self, obs, act, reward, obs_, done):
        self.obs_buf[self.ptr,:] = obs
        self.act_buf[self.ptr,:] = act
        self.reward_buf[self.ptr] = reward
        self.obs_next_buf[self.ptr,:] = obs_
        self.done_buf[self.ptr] = done
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.ptr = 0
            self.buffer_filled = True
    
    def get(self, batch_size):
        if self.buffer_filled:
            idx = np.random.choice(self.buffer_size, batch_size)
        else:
            idx = np.random.choice(self.ptr, batch_size)
        return self.obs_buf[idx,:], self.act_buf[idx,:], self.reward_buf[idx], self.obs_next_buf[idx,:], self.done_buf[idx]