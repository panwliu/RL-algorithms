import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import nns
import time
import envs
import argparse, sys
import itertools, copy
import mpi_tools

class ReplayBuffer:
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

class rlSAC:
    def __init__(self, ac:nns.MLPActorCritic3, ac_targ:nns.MLPActorCritic3) -> None:
        self.ac, self.ac_targ = ac, ac_targ
        self.gamma, self.batch_size, self.alpha = 0.99, 256, 0.2

        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        self.optim_pi = torch.optim.Adam(self.ac.pi.parameters(), lr=0.001)
        self.optim_q = torch.optim.Adam(self.q_params, lr=0.001)

    def action(self, obs, deterministic=False):
        return self.ac.act(torch.as_tensor(obs, dtype=torch.float32), deterministic)

    def train(self, buffer:ReplayBuffer):
        obs, acts, rewards, obs_, done = buffer.get(self.batch_size)

        obs = torch.tensor(obs, dtype=torch.float32)
        acts = torch.tensor(acts, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        obs2 = torch.tensor(obs_, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q1 = self.ac.q1(obs,acts)
        q2 = self.ac.q2(obs,acts)

        with torch.no_grad():
            # Target actions come from *current* policy
            acts2, logp_act2 = self.ac.pi(obs2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(obs2, acts2)
            q2_pi_targ = self.ac_targ.q2(obs2, acts2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rewards + self.gamma * (1 - done) * (q_pi_targ - self.alpha * logp_act2)

        self.optim_q.zero_grad()
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        loss_q.backward()
        mpi_tools.mpi_avg_grads(self.ac.q1)
        mpi_tools.mpi_avg_grads(self.ac.q2)
        self.optim_q.step()

        for p in self.q_params:
            p.requires_grad = False

        pi, logp_pi = self.ac.pi(obs)
        q1_pi = self.ac.q1(obs, pi)
        q2_pi = self.ac.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        self.optim_pi.zero_grad()
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        loss_pi.backward()
        mpi_tools.mpi_avg_grads(self.ac.pi)
        self.optim_pi.step()

        for p in self.q_params:
            p.requires_grad = True

        polyak = 0.995
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # ------------ CartPole ------------
    # parser.add_argument('--env', type=str, default='CartPole-v1')
    # parser.add_argument('--memory_size', type=int, default=2000)
    # parser.add_argument('--hid', type=list, default=[128,64])
    # parser.add_argument('--lr_a', type=float, default=1e-2)
    # parser.add_argument('--lr_c', type=float, default=1e-2)
    # parser.add_argument('--train_a_itrs', type=int, default=20)
    # parser.add_argument('--train_c_itrs', type=int, default=3)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--save_freq', type=int, default=50)
    # parser.add_argument('--eval', default=False, action='store_true')
    # parser.add_argument('--eval_path', type=str, default='')
    # parser.add_argument('--play_speed', type=int, default=1)
    # parser.add_argument('--num_procs', type=int, default=1)
    # ------------ Walker2D-v0/1 ------------
    parser.add_argument('--env', type=str, default='Walker2D-v1')
    parser.add_argument('--memory_size', type=int, default=5000)
    parser.add_argument('--hid', type=list, default=[128,64])
    parser.add_argument('--lr_a', type=float, default=1e-3)
    parser.add_argument('--lr_c', type=float, default=1e-3)
    parser.add_argument('--total_steps', type=int, default=25000)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_path', type=str, default='')
    parser.add_argument('--play_speed', type=int, default=1)
    parser.add_argument('--num_procs', type=int, default=4)
    args = parser.parse_args()

    mpi_tools.mpi_fork(args.num_procs)
    proc_id = mpi_tools.proc_id()

    mpi_tools.setup_pytorch_for_mpi()
    
    # Random seed
    seed = 0
    seed += 10000 * proc_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.env_by_name(args.env)
    print('memory_size ', args.memory_size, flush=True)

    buffer = ReplayBuffer(env.obs_dim, env.act_dim, args.memory_size)
    ac = nns.MLPActorCritic3(env.obs_dim, env.act_dim, args.hid, torch.nn.ReLU)
    # ac_targ = nns.MLPActorCritic3(env.obs_dim, env.act_dim, args.hid, torch.nn.ReLU)
    mpi_tools.sync_params(ac)
    ac_targ = copy.deepcopy(ac)
    agent = rlSAC(ac, ac_targ)
    
    k_ep, k_ep_step = 0, 0
    reward_total = 0
    obs = env.reset()
    for t in range(args.total_steps):
        # if k_ep > 300:
        #     env.render()
        
        if t > 2000:
            act = agent.action(obs)
        else:
            act = np.random.uniform(low=-1.0, high=1.0, size=env.act_dim)
        obs_, reward, done = env.step(act)

        buffer.store(obs, act, reward, obs_, done)

        obs = obs_
        reward_total += reward

        k_ep_step += 1
        if done:
            if k_ep % 10 == 0 and proc_id == 0:
                print(k_ep,"Episode finished after {} timesteps".format(k_ep_step), ", reward is {}".format(reward_total), flush=True)
            obs = env.reset()
            k_ep += 1
            k_ep_step = 0
            reward_total = 0

        if t > 1000:
            agent.train(buffer)
        
        # if t%2000==0:
        #     mpi_tools.sync_params(ac)
        #     mpi_tools.sync_params(ac_targ)

    if proc_id == 0:
        print('--------- test policy ---------')
        k_ep, k_ep_step = 0, 0
        reward_total = 0
        obs = env.reset()
        for t in range(1000):
            env.render()
            act = agent.action(obs,True)
            obs, reward, done = env.step(act)
            reward_total += reward
            k_ep_step += 1
            if done or k_ep_step==999:
                print(k_ep,"Episode finished after {} timesteps".format(k_ep_step), ", reward is {}".format(reward_total), flush=True)
                obs = env.reset()
                k_ep += 1
                k_ep_step = 0
                reward_total = 0
