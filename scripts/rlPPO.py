import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from functools import partial
import envs




class rlPPO_old():
    def __init__(self, n_states, n_actions, continuous):
        self.ns, self.na, self.continuous = n_states, n_actions, continuous
        self.gamma = 0.9

        self.transition_memory_idx, self.transition_memory_size = -1, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_size, self.ns + self.na + 1 + self.ns)) if self.continuous else np.zeros((self.transition_memory_size, self.ns + 1 + 1 + self.ns))

        if self.continuous:
            self.actor = ActorNetwork(in_features=self.ns, out_features=2*self.na)
            self.critic = CriticNetwork(in_features=self.ns)
            self.actor_old = ActorNetwork(in_features=self.ns, out_features=2*self.na)
            self.critic_target = CriticNetwork(in_features=self.ns)
        else:
            self.actor = ActorNetwork(in_features=self.ns, out_features=self.na)
            self.critic = CriticNetwork(in_features=self.ns)
            self.actor_old = ActorNetwork(in_features=self.ns, out_features=self.na)
            self.critic_target = CriticNetwork(in_features=self.ns)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # self.actor_old_reload_cnt = 5
        self.critic_target_reload_cnt = 1

        self.optim_a = torch.optim.Adam(self.actor.parameters())
        self.optim_c = torch.optim.Adam(self.critic.parameters())

        self.writer = SummaryWriter( log_dir='./log/PPO/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) )
        x = torch.rand(1,self.ns)
        self.writer.add_graph(self.actor, input_to_model=x)
        self.writer.add_graph(self.critic, input_to_model=x)

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        if self.continuous:
            mu_sigma = self.actor(x).detach()
            torch.clamp_min_(mu_sigma[:,self.na:], 0.05)
            a = torch.normal(mean=mu_sigma[0,:self.na], std=mu_sigma[0,self.na:]).view(-1).numpy()
        else:
            # a_prob = F.softmax( self.actor(x).detach(), dim=1).detach().numpy()
            a_prob = F.log_softmax(self.actor(x).detach(), dim=1).exp().numpy() # log_softmax is faster and has better numerical properties
            a = np.random.choice(self.na, 1, p=a_prob.squeeze())

        return a

    def store_transition(self, state, action, reward, state_):
        self.transition_memory_idx += 1
        if self.transition_memory_idx == self.transition_memory_size:
            self.transition_memory_idx = 0
            self.transition_memory_filled = True

        self.transition_memory[self.transition_memory_idx, :] = np.hstack((state,action,reward,state_))

    def train(self):
        # if self.transition_memory_idx % self.actor_old_reload_cnt == 0:
        #     self.actor_old.load_state_dict(self.actor.state_dict())
        if self.transition_memory_idx % self.critic_target_reload_cnt == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())

        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        if self.continuous:
            memory = self.transition_memory[sample_idx,:]
            states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
            actions = memory[:,self.ns:self.ns+self.na]
            rewards = torch.tensor(memory[:,self.ns+self.na], dtype=torch.float32).view(-1,1)
            states_ = torch.tensor(memory[:,self.ns+self.na+1:], dtype=torch.float32)

            self.optim_c.zero_grad()
            Q_target = rewards + self.gamma*self.critic_target(states_).detach()
            loss_c = F.mse_loss(self.critic(states), Q_target)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 1.0)
            self.optim_c.step()

            state = torch.tensor( self.transition_memory[self.transition_memory_idx,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
            action = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns:self.ns+self.na], dtype=torch.float32 ).view(-1,self.na)
            reward = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns+self.na], dtype=torch.float32 )
            state_ = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns+self.na+1:], dtype=torch.float32 ).view(-1,self.ns)

            self.optim_a.zero_grad()
            advantage = reward + self.gamma*self.critic(state_).detach() - self.critic(state).detach()      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
            mu_sigma, mu_sigma_old = self.actor(state), self.actor_old(state).detach()
            torch.clamp_min_(mu_sigma[:,self.na:], 0.05)
            torch.clamp_min_(mu_sigma_old[:,self.na:], 0.05)
            dist, dist_old = torch.distributions.Normal(mu_sigma[:,:self.na], mu_sigma[:,self.na:]), torch.distributions.Normal(mu_sigma_old[:,:self.na], mu_sigma_old[:,self.na:])
            ratio = ( dist.log_prob(action) - dist_old.log_prob(action) ).exp()
            if ratio.detach()>10.0 and advantage<0:
                print("Early terminate training!")
                return
            loss_a = -torch.min( advantage*ratio, advantage*torch.clamp(ratio, 1-0.2, 1+0.2) )
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), 1.0)
            if np.isnan(self.actor.l1.weight.grad.detach().numpy()).any():
                print("grad nan!!")
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.optim_a.step()

        else:
            memory = self.transition_memory[sample_idx,:]
            states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
            actions = memory[:,self.ns]
            rewards = torch.tensor(memory[:,self.ns+1], dtype=torch.float32).view(-1,1)
            states_ = torch.tensor(memory[:,self.ns+2:], dtype=torch.float32)

            self.optim_c.zero_grad()
            Q_target = rewards + self.gamma*self.critic_target(states_).detach()
            loss_c = F.mse_loss(self.critic(states), Q_target)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 1.0)
            self.optim_c.step()

            state = torch.tensor(self.transition_memory[self.transition_memory_idx,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
            action = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns].astype(int), dtype=torch.int64).view(-1)
            reward = self.transition_memory[self.transition_memory_idx,self.ns+1]
            state_ = torch.tensor(self.transition_memory[self.transition_memory_idx,self.ns+2:], dtype=torch.float32 ).view(-1,self.ns)

            self.optim_a.zero_grad()
            advantage = reward + self.gamma*self.critic(state_).detach() - self.critic(state).detach()      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
            ratio = ( F.log_softmax(self.actor(state), dim=1) - F.log_softmax(self.actor_old(state).detach(), dim=1) ).exp()
            ratio = torch.sum( ratio * F.one_hot(action, self.na), dim=1)
            loss_a = -torch.min( advantage*ratio, advantage*torch.clamp(ratio, 1-0.2, 1+0.2) )
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), 1.0)
            self.actor_old.load_state_dict(self.actor.state_dict())
            self.optim_a.step()

    def write_reward(self, k_ep, reward):
        self.writer.add_scalar("Reward", reward, k_ep)


class ActorNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ActorNetwork,self).__init__()

        n_in, n_out = in_features, out_features

        self.l1 = torch.nn.Linear(in_features=n_in, out_features=32)
        self.l2 = torch.nn.Linear(in_features=32, out_features=n_out)

    def forward(self, x):
        l1_o = F.relu( self.l1(x) )
        l2_o = self.l2(l1_o)        # output logits, rather than probabilities

        return l2_o

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

class rlPPO:
    def __init__(self, n_states, n_actions):
        self.ns, self.na, self.gamma = n_states, n_actions, 0.99
        self.GAE, self.lam = False, 1.0

        self.transition_memory = np.empty(shape=(0,self.ns+self.na+1), dtype=np.float32)
        self.reward_to_go, self.adv = np.array([]), np.array([])
        
        self.actor = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)
        self.dist = partial( torch.distributions.Normal, scale=0.3 )
        self.actor_old = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.lr_a, self.lr_c = 1e-2, 1e-2
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.writer = SummaryWriter( log_dir='./log/PPO/' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) )
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
        actions = torch.tensor(actions, dtype=torch.float32).view(-1,self.na)
        rewards_cum = torch.tensor(rewards_cum, dtype=torch.float32)

        if self.GAE:
            advantages = torch.tensor(self.adv, dtype=torch.float32)
        else:
            baselines = self.critic(states).detach().view(-1)
            advantages = rewards_cum - baselines

        logp_old = self.dist(loc=self.actor_old(states).detach()).log_prob(actions).sum(axis=-1)
        for _ in range(10):
            logp = self.dist(loc=self.actor(states)).log_prob(actions).sum(axis=-1)
            ratios = (logp - logp_old).exp()
            loss_a = -torch.min( advantages*ratios, advantages*torch.clamp(ratios, 1-0.2, 1+0.2) ).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            self.optimizer_a.step()
        self.actor_old.load_state_dict(self.actor.state_dict())
        
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
    env = envs.env_by_name("CartPole-v1")
    agent = rlPPO(n_states=4, n_actions=1)

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