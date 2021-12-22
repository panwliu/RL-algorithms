import torch

class BufferBase:
    def store(self):
        raise NotImplementedError
    def get(self):
        raise NotImplementedError

# for policy gradient based algorithms, e.g. reinforce, vpg, ppo
class ExpBuffer(BufferBase):
    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros((buffer_size,obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((buffer_size,act_dim), dtype=torch.float32)
        self.reward_buf = torch.zeros(buffer_size, dtype=torch.float32)
        self.obs_next_buf = torch.zeros((buffer_size,obs_dim), dtype=torch.float32)

        self.reward_to_go_buf = torch.zeros(buffer_size, dtype=torch.float32)
        self.logp_buff = torch.zeros(buffer_size, dtype=torch.float32)
        
        self.gamma = gamma
        self.ptr, self.traj_start_idx = 0, 0

    def store(self, obs, act, reward, obs_next, logp):
        self.obs_buf[self.ptr,:] = torch.tensor(obs, dtype=torch.float32)
        self.act_buf[self.ptr,:] = torch.tensor(act, dtype=torch.float32)
        self.reward_buf[self.ptr] = reward
        self.obs_next_buf[self.ptr,:] = torch.tensor(obs_next, dtype=torch.float32)
        self.logp_buff[self.ptr] = torch.tensor(logp, dtype=torch.float32)
        
        self.ptr += 1
    
    def finish_traj(self, last_val=0):
        rewards = self.reward_buf[self.traj_start_idx:self.ptr]
        r_cum = torch.zeros_like(rewards)

        r_cum[-1] = rewards[-1] + last_val
        for k in reversed(range(len(rewards)-1)):
            r_cum[k] = rewards[k] + self.gamma*r_cum[k+1]

        self.reward_to_go_buf[self.traj_start_idx:self.ptr] = r_cum

        self.traj_start_idx = self.ptr

    def get(self):
        self.ptr, self.traj_start_idx = 0, 0
        return self.obs_buf, self.act_buf, self.reward_buf, self.obs_next_buf, self.reward_to_go_buf, self.logp_buff


# for value function based RL algorithms, e.g. DQN
class ReplayBuffer(BufferBase):
    def __init__(self, obs_dim, act_dim, buffer_size):
        self.buffer_size = buffer_size
        self.obs_buf = torch.zeros((buffer_size,obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((buffer_size,act_dim), dtype=torch.float32)
        self.reward_buf = torch.zeros(buffer_size, dtype=torch.float32)
        self.obs_next_buf = torch.zeros((buffer_size,obs_dim), dtype=torch.float32)
        self.done_buf = torch.zeros(buffer_size, dtype=torch.float32)
        
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
            idx = torch.randint(self.buffer_size, (batch_size,))
        else:
            idx = torch.randint(self.ptr, (batch_size,))
        return self.obs_buf[idx,:], self.act_buf[idx,:], self.reward_buf[idx], self.obs_next_buf[idx,:], self.done_buf[idx]