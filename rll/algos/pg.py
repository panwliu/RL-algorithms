import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from functools import partial
import envs

class PGNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PGNetwork,self).__init__()

        n_in, n_out = in_features, out_features

        self.l1 = torch.nn.Linear(in_features=n_in, out_features=32)
        self.l2 = torch.nn.Linear(in_features=32, out_features=n_out)

    def forward(self, x):
        l1_o = F.relu( self.l1(x) )
        l2_o = self.l2(l1_o)        # output logits, rather than probabilities

        return l2_o

# ------------ Policy Gradient ------------
# REINFORCE: gamma = 1, i.e. no discounting, and no reward-to-go, i.e. constant R for entire traj
class rlPG:
    def __init__(self, n_states, n_actions):
        self.ns, self.na = n_states, n_actions
        self.gamma = 0.99

        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)
        self.reward_to_go = np.array([])
        
        self.net = PGNetwork(in_features=self.ns, out_features=self.na)
        self.dist = partial( torch.distributions.Categorical )

        self.lr = 1e-2
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.writer = SummaryWriter( log_dir='./log/PG/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) )
        x = torch.rand(1,self.ns)
        self.writer.add_graph(self.net, input_to_model=x)

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        # a_prob = F.log_softmax( self.net(x).detach(), dim=1 ).exp().numpy()     # TODO: change to log_softmax
        # a = np.random.choice(self.na, size=1, p=a_prob.squeeze())
        a = self.dist(logits = self.net(x).detach()).sample().numpy()

        return a

    def store_transition(self, state, action, reward):
        transition = np.hstack((state, action, reward)).reshape(1,-1)
        self.transition_memory = np.append(self.transition_memory, transition, axis=0)

    def cumulative_reward(self):
        rewards = self.transition_memory[len(self.reward_to_go):, -1]
        r_cum = np.zeros_like(rewards)

        r_cum[-1] = rewards[-1]
        for k in reversed(range(len(rewards)-1)):
            r_cum[k] = rewards[k] + self.gamma*r_cum[k+1]

        self.reward_to_go = np.append(self.reward_to_go, r_cum)

    def train(self):
        states = self.transition_memory[:,:self.ns]
        actions = self.transition_memory[:,self.ns]
        rewards = self.transition_memory[:,self.ns+1]
        rewards_cum = self.reward_to_go
        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)
        self.reward_to_go = np.array([])

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards_cum = torch.tensor(rewards_cum, dtype=torch.float32)

        for _ in range(1):      # better performance, but should notice the off-policy after 1st run
            # cross_entropy = F.cross_entropy(self.net(states), actions, reduction='none')
            # loss = torch.mean( cross_entropy * rewards_cum )
            logp = self.dist(logits=self.net(states)).log_prob(actions)
            loss = - (logp * rewards_cum).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def write_reward(self, k_epoch, reward, ep_len):
        self.writer.add_scalar("Reward", reward, k_epoch)
        self.writer.add_scalar("Ep_len", ep_len, k_epoch)


# ------------ Vanilla Policy Gradient Discrete ------------
# VPG = PG + Advantage
# Two ways for baseline selection
#   1. average cross sampled trajs
#   2. Use critic network to approximate value function. Two ways to fit critic network:
#       1. Monte Carlo:
#       2. Bootstrap: more unstable
class CriticNetwork(torch.nn.Module):
    def __init__(self, in_features):
        super(CriticNetwork,self).__init__()

        n_in = in_features

        self.l1 = torch.nn.Linear(in_features=n_in, out_features=32)
        self.l2 = torch.nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        l1_o = F.relu( self.l1(x) )
        l2_o = self.l2(l1_o)

        return l2_o

class rlVPGv1:
    def __init__(self, n_states, n_actions):
        self.ns, self.na, self.gamma = n_states, n_actions, 0.99
        self.GAE, self.lam = False, 1.0

        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)
        self.reward_to_go, self.adv = np.array([]), np.array([])
        
        self.actor = PGNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)
        self.dist = partial( torch.distributions.Categorical )

        self.lr = 1e-2
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.writer = SummaryWriter( log_dir='./log/PG/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) )
        x = torch.rand(1,self.ns)
        self.writer.add_graph(self.actor, input_to_model=x)
        self.writer.add_graph(self.critic, input_to_model=x)

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        a = self.dist(logits = self.actor(x).detach()).sample().numpy()
        return a

    def store_transition(self, state, action, reward):
        transition = np.hstack((state, action, reward)).reshape(1,-1)
        self.transition_memory = np.append(self.transition_memory, transition, axis=0)

    def cumulative_reward(self):
        states = self.transition_memory[len(self.reward_to_go):,:self.ns]
        rewards = self.transition_memory[len(self.reward_to_go):, -1]
        
        vals = self.critic(torch.tensor(states, dtype=torch.float32)).detach().view(-1).numpy()
        vals = np.append(vals, 0)
        deltas = rewards + self.gamma*vals[1:] - vals[:-1] 
        
        r_cum = np.zeros_like(rewards)
        adv = np.zeros_like(rewards)

        r_cum[-1] = rewards[-1]
        adv[-1] = deltas[-1]
        for k in reversed(range(len(rewards)-1)):
            r_cum[k] = rewards[k] + self.gamma*r_cum[k+1]
            adv[k] = deltas[k] + self.gamma*self.lam*adv[k+1]

        # adv = ( adv - np.mean(adv) ) / np.std(adv)

        self.reward_to_go = np.append(self.reward_to_go, r_cum)
        self.adv = np.append(self.adv, adv)

    def train(self):
        states = self.transition_memory[:,:self.ns]
        actions = self.transition_memory[:,self.ns]
        rewards = self.transition_memory[:,self.ns+1]
        rewards_cum = self.reward_to_go

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards_cum = torch.tensor(rewards_cum, dtype=torch.float32)

        if self.GAE:
            advantages = torch.tensor(self.adv, dtype=torch.float32)
        else:
            baselines = self.critic(states).detach().view(-1)
            advantages = rewards_cum - baselines

        for _ in range(3):      # better performance, but should notice the off-policy after 1st run
            logp = self.dist(logits=self.actor(states)).log_prob(actions)
            loss_a = - (logp * advantages).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
        
        for _ in range(3):
            self.optimizer_c.zero_grad()
            loss_c = F.mse_loss(self.critic(states), rewards_cum.view(-1,1))
            loss_c.backward()
            self.optimizer_c.step()

        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)
        self.reward_to_go = np.array([])
        self.adv = np.array([])
    
    def write_reward(self, k_epoch, reward, ep_len):
        self.writer.add_scalar("Reward", reward, k_epoch)
        self.writer.add_scalar("Ep_len", ep_len, k_epoch)


# ------------ Vanilla Policy Gradient ------------
class rlVPG:
    def __init__(self, n_states, n_actions):
        self.ns, self.na, self.gamma = n_states, n_actions, 0.99
        self.GAE, self.lam = False, 1.0

        self.transition_memory = np.empty(shape=(0,self.ns+self.na+1), dtype=np.float32)
        self.reward_to_go, self.adv = np.array([]), np.array([])
        
        self.actor = PGNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)
        self.dist = partial( torch.distributions.Normal, scale=torch.tensor([0.3]) )

        self.lr_a, self.lr_c = 1e-2, 1e-2
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.writer = SummaryWriter( log_dir='./log/PG/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) )
        x = torch.rand(1,self.ns)
        self.writer.add_graph(self.actor, input_to_model=x)
        self.writer.add_graph(self.critic, input_to_model=x)

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        a = self.dist(loc = self.actor(x).detach()).sample().view(-1).numpy()
        return a

    def store_transition(self, state, action, reward):
        transition = np.hstack((state, action, reward)).reshape(1,-1)
        self.transition_memory = np.append(self.transition_memory, transition, axis=0)

    def cumulative_reward(self):
        states = self.transition_memory[len(self.reward_to_go):,:self.ns]
        rewards = self.transition_memory[len(self.reward_to_go):, -1]
        
        vals = self.critic(torch.tensor(states, dtype=torch.float32)).detach().view(-1).numpy()
        vals = np.append(vals, 0)
        deltas = rewards + self.gamma*vals[1:] - vals[:-1] 
        
        r_cum = np.zeros_like(rewards)
        adv = np.zeros_like(rewards)

        r_cum[-1] = rewards[-1]
        adv[-1] = deltas[-1]
        for k in reversed(range(len(rewards)-1)):
            r_cum[k] = rewards[k] + self.gamma*r_cum[k+1]
            adv[k] = deltas[k] + self.gamma*self.lam*adv[k+1]

        # adv = ( adv - np.mean(adv) ) / np.std(adv)

        self.reward_to_go = np.append(self.reward_to_go, r_cum)
        self.adv = np.append(self.adv, adv)

    def train(self):
        states = self.transition_memory[:,:self.ns]
        actions = self.transition_memory[:,self.ns]
        rewards = self.transition_memory[:,self.ns+1]
        rewards_cum = self.reward_to_go

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32).view(-1,1)
        rewards_cum = torch.tensor(rewards_cum, dtype=torch.float32)

        if self.GAE:
            advantages = torch.tensor(self.adv, dtype=torch.float32)
        else:
            baselines = self.critic(states).detach().view(-1)
            advantages = rewards_cum - baselines

        for _ in range(3):
            logp = self.dist(loc=self.actor(states)).log_prob(actions).sum(axis=-1)
            loss_a = - (logp/100 * advantages).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
        
        for _ in range(3):
            self.optimizer_c.zero_grad()
            loss_c = F.mse_loss(self.critic(states), rewards_cum.view(-1,1))
            loss_c.backward()
            self.optimizer_c.step()

        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)
        self.reward_to_go = np.array([])
        self.adv = np.array([])
    
    def write_reward(self, k_epoch, reward, ep_len):
        self.writer.add_scalar("Reward", reward, k_epoch)
        self.writer.add_scalar("Ep_len", ep_len, k_epoch)


if __name__ == "__main__":
    # env = envs.env_by_name("CartPole-v0")
    # # agent = rlPGv1(n_states=4, n_actions=2)
    # agent = rlVPGv1(n_states=4, n_actions=2)
    env = envs.env_by_name("CartPole-v1")
    agent = rlVPG(n_states=4, n_actions=1)

    for k_epoch in range(100):
        state = env.reset()

        k_ep = 0
        while True:

            # if k_epoch > 50:
            #     env.render()

            action = agent.action(state)
            state_, reward, done = env.step(action)

            agent.store_transition(state, action, reward)

            state = state_

            if done:
                k_ep += 1
                agent.cumulative_reward()
                if agent.transition_memory.shape[0]>5000: 
                    break
                env.reset()
        
        print( 'Epoch: %3d \t return: %.3f \t ep_len: %.3f' 
                %(k_epoch, np.mean(agent.reward_to_go), agent.transition_memory.shape[0]/k_ep) )
        agent.write_reward(k_epoch, np.mean(agent.reward_to_go), agent.transition_memory.shape[0]/k_ep)
        agent.train()