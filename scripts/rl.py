import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from envs import CartpoleEnv, CartpoleEnv_continuous
import ray

# ------------ DQN ------------ 
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        ns, na = 4, 2

        self.l1 = torch.nn.Linear(in_features=ns, out_features=100)
        self.l2 = torch.nn.Linear(in_features=100, out_features=20)
        self.l3 = torch.nn.Linear(in_features=20, out_features=na)

    def forward(self, x):
        l1_o = F.relu( self.l1(x) )
        l2_o = F.relu( self.l2(l1_o) )
        l3_o = self.l3(l2_o)

        return l3_o

class rlDQN():
    def __init__(self, n_states=4, n_actions=2):
        self.ns, self.na = n_states, n_actions
        self.epsilon = 1
        self.epsilon_cnt = 0

        self.transition_memory_idx, self.transition_memory_capacity = 0, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_capacity, self.ns + 1 + 1 + self.ns))

        self.net = DQN()
        self.net_target = DQN()
        self.net_target.load_state_dict(self.net.state_dict())
        self.cnt = 0

        self.lr, self.gamma = 1e-3, 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def action(self, x:np.ndarray):
        self.epsilon_cnt+=1
        if self.epsilon_cnt%50==0 and self.epsilon>0.1:
            self.epsilon -= 0.02

        if np.random.rand() < self.epsilon:     # explore
            a = np.random.choice(self.na, 1)
        else:                                   # exploit
            x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
            Q = self.net(x).detach().numpy()
            a = np.argmax(Q)
        
        return a

    def store_transition(self, state, action, reward, state_):
        if self.transition_memory_idx >= self.transition_memory_capacity:
            self.transition_memory_idx = 0
            self.transition_memory_filled = True

        self.transition_memory[self.transition_memory_idx, :] = np.hstack((state,action,reward,state_))
        self.transition_memory_idx += 1

    def train(self):
        self.cnt += 1
        if self.cnt%100 == 0:
            self.net_target.load_state_dict(self.net.state_dict())

        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_capacity, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        memory = self.transition_memory[sample_idx,:]

        states = memory[:, 0:self.ns].astype(np.float32)        # numpy default np.float64, while pytorch tensor default to be np.float32
        actions = memory[:, self.ns].astype(np.int)
        rewards = memory[:, self.ns+1].astype(np.float32)
        states_ = memory[:, self.ns+2:].astype(np.float32)

        Q_pred = self.net( torch.from_numpy(states) )
        
        Q_pred_next = self.net_target( torch.from_numpy(states_) ).detach()
        Q_target = self.net( torch.from_numpy(states) ).detach()

        Q_target[np.arange(self.batch_size), actions] = torch.from_numpy(rewards) + self.gamma*torch.max(Q_pred_next,dim=1)[0]
        # Q_target[np.arange(self.batch_size), actions] += 0.05 *(
        #      torch.from_numpy(rewards) + self.gamma*torch.max(Q_pred_next,dim=1)[0] - Q_target[np.arange(self.batch_size), actions] )

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_pred, Q_target)
        loss.backward()
        self.optimizer.step()
            

# ------------ PG ------------ 
class PGNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(PGNetwork,self).__init__()

        n_in, n_out = in_features, out_features

        self.l1 = torch.nn.Linear(in_features=n_in, out_features=100)
        self.l2 = torch.nn.Linear(in_features=100, out_features=50)
        self.l3 = torch.nn.Linear(in_features=50, out_features=n_out)

    def forward(self, x):
        l1_o = F.relu( self.l1(x) )
        l2_o = F.relu( self.l2(l1_o) )
        l3_o = self.l3( l2_o )

        return l3_o

class rlPG:
    def __init__(self, n_states, n_actions):
        self.ns, self.na = n_states, n_actions
        self.gamma = 0.99

        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)
        
        self.net = PGNetwork(in_features=self.ns, out_features=self.na)

        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        a_prob = F.softmax( self.net(x).detach() ).detach().numpy()     # TODO: change to log_softmax

        a = np.random.choice(self.na, size=1, p=a_prob.squeeze())

        return a

    def store_transition(self, state, action, reward):
        transition = np.hstack((state, action, reward)).reshape(1,-1)
        self.transition_memory = np.append(self.transition_memory, transition, axis=0)

    def cumulative_reward(self, rewards):
        r_cum = np.zeros_like(rewards)

        r_cum[-1] = rewards[-1]
        for k in reversed(range(len(rewards)-1)):
            r_cum[k] = rewards[k] + self.gamma*r_cum[k+1]

        return r_cum

    def train(self):
        states = self.transition_memory[:,:self.ns]
        actions = self.transition_memory[:,self.ns]
        rewards = self.transition_memory[:,self.ns+1]
        rewards_cum = self.cumulative_reward(rewards)
        self.transition_memory = np.empty(shape=(0,self.ns+1+1), dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards_cum = torch.tensor(rewards_cum, dtype=torch.float32)

        cross_entropy = F.cross_entropy(self.net(states), actions, reduction='none')
        loss = torch.mean( cross_entropy * rewards_cum )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ------------ A2C ------------ 
class ActorNetwork(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ActorNetwork,self).__init__()

        n_in, n_out = in_features, out_features

        self.l1 = torch.nn.Linear(in_features=n_in, out_features=100)
        self.l2 = torch.nn.Linear(in_features=100, out_features=50)
        self.l3 = torch.nn.Linear(in_features=50, out_features=n_out)

    def forward(self, x):
        l1_o = F.relu( self.l1(x) )
        l2_o = F.relu( self.l2(l1_o) )
        l3_o = self.l3( l2_o )

        return l3_o

class CriticNetwork(torch.nn.Module):
    def __init__(self, in_features):
        super(CriticNetwork,self).__init__()

        n_in = in_features

        self.l1 = torch.nn.Linear(in_features=n_in, out_features=100)
        self.l2 = torch.nn.Linear(in_features=100, out_features=20)
        self.l3 = torch.nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        l1_out = F.relu( self.l1(x) )
        l2_out = F.relu( self.l2(l1_out) )
        l3_out = self.l3( l2_out )

        return l3_out

class rlA2C():
    def __init__(self, n_states, n_actions):
        self.ns, self.na = n_states, n_actions
        self.gamma = 0.9

        self.transition_memory_idx, self.transition_memory_size = -1, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_size, self.ns + 1 + 1 + self.ns))

        self.actor = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)

        self.optim_a = torch.optim.Adam(self.actor.parameters())
        self.optim_c = torch.optim.Adam(self.critic.parameters())

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        a_prob = F.softmax( self.actor(x).detach(), dim=1).detach().numpy()

        a = np.random.choice(self.na, 1, p=a_prob.squeeze())

        return a

    def store_transition(self, state, action, reward, state_):
        self.transition_memory_idx += 1
        if self.transition_memory_idx == self.transition_memory_size:
            self.transition_memory_idx = 0
            self.transition_memory_filled = True

        self.transition_memory[self.transition_memory_idx, :] = np.hstack((state,action,reward,state_))

    def train(self):
        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        memory = self.transition_memory[sample_idx,:]
        states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
        actions = memory[:,self.ns]
        rewards = torch.tensor(memory[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor(memory[:,self.ns+2:], dtype=torch.float32)

        self.optim_c.zero_grad()     
        Q_target = rewards + self.gamma*self.critic(states_).detach()
        loss = F.mse_loss(self.critic(states), Q_target)
        loss.backward()
        self.optim_c.step()

        state = torch.tensor(self.transition_memory[self.transition_memory_idx,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
        action = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns].astype(int), dtype=torch.int64).view(-1)
        reward = self.transition_memory[self.transition_memory_idx,self.ns+1]
        state_ = torch.tensor(self.transition_memory[self.transition_memory_idx,self.ns+2:], dtype=torch.float32 ).view(-1,self.ns)

        self.optim_a.zero_grad()
        advantage = reward + self.gamma*self.critic(state_).detach() - self.critic(state).detach()      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
        cross_entropy = F.cross_entropy(self.actor(state), action, reduction='none')
        loss = advantage * cross_entropy        # loss = Q(s,a) * cross_entropy for AC
        loss.backward()
        self.optim_a.step()



# ------------ A3C ------------
@ray.remote
class Worker():
    def __init__(self, agent_global:rlA2C, n_states, n_actions):
        self.agent_global, self.ns, self.na = agent_global, n_states, n_actions
        self.gamma = 0.9

        self.transition_memory_idx, self.transition_memory_size = -1, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_size, self.ns + 1 + 1 + self.ns))

        self.actor = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)

        self.optim_a = torch.optim.Adam(self.actor.parameters())
        self.optim_c = torch.optim.Adam(self.critic.parameters())

        self.env = CartpoleEnv("./models/cartpole.xml")

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        a_prob = F.softmax( self.actor(x).detach(), dim=1).detach().numpy()

        a = np.random.choice(self.na, 1, p=a_prob.squeeze())

        return a

    def store_transition(self, state, action, reward, state_):
        self.transition_memory_idx += 1
        if self.transition_memory_idx == self.transition_memory_size:
            self.transition_memory_idx = 0
            self.transition_memory_filled = True

        self.transition_memory[self.transition_memory_idx, :] = np.hstack((state,action,reward,state_))

    def train(self):
        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        memory = self.transition_memory[sample_idx,:]
        states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
        actions = memory[:,self.ns]
        rewards = torch.tensor(memory[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor(memory[:,self.ns+2:], dtype=torch.float32)

        self.optim_c.zero_grad()
        Q_target = rewards + self.gamma*self.critic(states_).detach()
        loss_c = F.mse_loss(self.critic(states), Q_target)
        loss_c.backward()
        # self.optim_c.step()

        state = torch.tensor(self.transition_memory[self.transition_memory_idx,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
        action = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns].astype(int), dtype=torch.int64).view(-1)
        reward = self.transition_memory[self.transition_memory_idx,self.ns+1]
        state_ = torch.tensor(self.transition_memory[self.transition_memory_idx,self.ns+2:], dtype=torch.float32 ).view(-1,self.ns)

        self.optim_a.zero_grad()
        advantage = reward + self.gamma*self.critic(state_).detach() - self.critic(state).detach()      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
        cross_entropy = F.cross_entropy(self.actor(state), action, reduction='none')
        loss_a = advantage * cross_entropy        # loss = Q(s,a) * cross_entropy for AC
        loss_a.backward()
        # self.optim_a.step()

        self.agent_global.optim_c.zero_grad()
        for net1,net2 in zip(self.critic.named_parameters(),self.agent_global.critic.named_parameters()):
            net2[1].grad = net1[1].grad.clone()
        self.agent_global.optim_c.step()

        self.agent_global.optim_a.zero_grad()
        for net1,net2 in zip(self.actor.named_parameters(),self.agent_global.actor.named_parameters()):
            net2[1].grad = net1[1].grad.clone()
        self.agent_global.optim_a.step()

        if self.transition_memory_idx % 2 == 0:
            self.actor.load_state_dict(self.agent_global.actor.state_dict())
            self.critic.load_state_dict(self.agent_global.critic.state_dict())


    def work(self):
        for k_ep in range(200):
            state = self.env.reset()

            k_step = 0
            while True:
                k_step += 1

                # self.env.render()

                action = self.action(state)
                state_, reward, done = self.env.step(action)

                self.store_transition(state, action, reward, state_)

                if k_ep > 10:
                    self.train()

                state = state_

                if done:
                    print("Ep ",k_ep, " step=", k_step)
                    break



class rlA3C():
    def __init__(self, n_states, n_actions, n_workers):
        self.ns, self.na, self.nw = n_states, n_actions, n_workers

        self.agent_global = rlA2C(n_states=self.ns, n_actions=self.na)  # TODO: passing reference doesn't work

        workers = [Worker.remote(agent_global=self.agent_global, n_states=self.ns, n_actions=self.na) for _ in range(self.nw)]

        ids = [worker.work.remote() for worker in workers]
        
        ray.get(ids)



# ------------ PPO ------------
@ray.remote
class WorkerPPO():
    def __init__(self, agent_global:rlA2C, n_states, n_actions):
        self.agent_global, self.ns, self.na = agent_global, n_states, n_actions
        self.gamma = 0.9

        self.transition_memory_idx, self.transition_memory_size = -1, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_size, self.ns + 1 + 1 + self.ns))

        self.actor = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)
        self.actor_old = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optim_a = torch.optim.Adam(self.actor.parameters())
        self.optim_c = torch.optim.Adam(self.critic.parameters())

        self.env = CartpoleEnv("./models/cartpole.xml")

        self.writer = SummaryWriter(log_dir='./log/')
        x = torch.rand(1,self.ns)
        self.writer.add_graph(self.actor, input_to_model=x)
        self.writer.add_graph(self.critic, input_to_model=x)

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
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
        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        memory = self.transition_memory[sample_idx,:]
        states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
        actions = memory[:,self.ns]
        rewards = torch.tensor(memory[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor(memory[:,self.ns+2:], dtype=torch.float32)

        self.optim_c.zero_grad()
        Q_target = rewards + self.gamma*self.critic(states_).detach()
        loss_c = F.mse_loss(self.critic(states), Q_target)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 1.0)
        # self.optim_c.step()

        state = torch.tensor(self.transition_memory[self.transition_memory_idx,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
        action = torch.tensor( self.transition_memory[self.transition_memory_idx,self.ns].astype(int), dtype=torch.int64).view(-1)
        reward = self.transition_memory[self.transition_memory_idx,self.ns+1]
        state_ = torch.tensor(self.transition_memory[self.transition_memory_idx,self.ns+2:], dtype=torch.float32 ).view(-1,self.ns)

        self.optim_a.zero_grad()
        advantage = reward + self.gamma*self.critic(state_).detach() - self.critic(state).detach()      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
        # cross_entropy = F.cross_entropy(self.actor(state), action, reduction='none')
        # loss_a = advantage * cross_entropy        # loss = Q(s,a) * cross_entropy for AC

        # r_theta = torch.sum( ( F.softmax(self.actor(state), dim=1) / F.softmax(self.actor_old(state).detach(), dim=1) ) * F.one_hot(action, self.na), dim=1 )
        r_theta = ( F.log_softmax(self.actor(state), dim=1) - F.log_softmax(self.actor_old(state).detach(), dim=1) ).exp()
        r_theta = torch.sum( r_theta * F.one_hot(action, self.na), dim=1)
        loss_a = -torch.min( advantage*r_theta, advantage*torch.clamp(r_theta, 1-0.2, 1+0.2) )
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), 1.0)
        # self.optim_a.step()

        self.agent_global.optim_c.zero_grad()
        for net1,net2 in zip(self.critic.named_parameters(),self.agent_global.critic.named_parameters()):
            net2[1].grad = net1[1].grad.clone()
        self.agent_global.optim_c.step()

        self.agent_global.optim_a.zero_grad()
        for net1,net2 in zip(self.actor.named_parameters(),self.agent_global.actor.named_parameters()):
            net2[1].grad = net1[1].grad.clone()
        self.agent_global.optim_a.step()

        self.actor.load_state_dict(self.agent_global.actor.state_dict())
        self.critic.load_state_dict(self.agent_global.critic.state_dict())

        if self.transition_memory_idx % 5 == 0:
            self.actor_old.load_state_dict(self.actor.state_dict())



    def work(self):
        for k_ep in range(500):
            state = self.env.reset()

            k_step = 0
            while True:
                k_step += 1

                # self.env.render()

                action = self.action(state)
                state_, reward, done = self.env.step(action)

                self.store_transition(state, action, reward, state_)

                if k_ep > 10:
                    self.train()

                state = state_

                if done:
                    print("Ep ",k_ep, " step=", k_step)
                    self.writer.add_scalar("Rewards", k_step, k_ep)
                    break


# ------------ DPPO Discrete ------------
@ray.remote
class WorkerDPPODiscrete():
    def __init__(self, n_states, n_actions):
        self.ns, self.na = n_states, n_actions
        self.actor, self.critic = ActorNetwork(self.ns,self.na), CriticNetwork(self.ns)
        self.env = CartpoleEnv("./models/cartpole.xml")

        self.k_ep, self.k_step = 0, 0
        self.state = self.env.reset()

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        a_prob = F.log_softmax(self.actor(x).detach(), dim=1).exp().numpy()     # log_softmax is faster and has better numerical properties

        a = np.random.choice(self.na, 1, p=a_prob.squeeze())

        return a

    def work(self):
        
        self.k_step += 1

        # self.env.render()

        action = self.action(self.state)
        state_, reward, done = self.env.step(action)

        buffer = np.hstack((self.state,action,reward,state_))

        self.state = state_

        if done:
            print("Ep ",self.k_ep, " step=", self.k_step)
            self.k_ep += 1
            self.k_step = 0
            self.state = self.env.reset()

        return buffer

    def set_parameters(self, actor:ActorNetwork, critic:CriticNetwork):
        self.actor.load_state_dict(actor.state_dict())
        self.critic.load_state_dict(critic.state_dict())
                    


class rlDPPODiscrete():
    def __init__(self, n_states, n_actions, n_workers):
        self.ns, self.na, self.nw = n_states, n_actions, n_workers
        self.gamma, self.train_flag = 0.9, False

        self.transition_memory_idx, self.transition_memory_size = -1, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_size, self.ns + 1 + 1 + self.ns))
        self.transition_buffer = np.zeros((self.nw, self.ns + 1 + 1 + self.ns))
        
        self.actor = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.critic = CriticNetwork(in_features=self.ns)
        self.actor_old = ActorNetwork(in_features=self.ns, out_features=self.na)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optim_a = torch.optim.Adam(self.actor.parameters())
        self.optim_c = torch.optim.Adam(self.critic.parameters())

        self.workers = [WorkerDPPODiscrete.remote(self.ns, self.na) for _ in range(self.nw)]
        ids = [worker.set_parameters.remote(self.actor, self.critic) for worker in self.workers]
        ray.get(ids)


    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        # a_prob = F.softmax( self.actor(x).detach(), dim=1).detach().numpy()
        a_prob = F.log_softmax(self.actor(x).detach(), dim=1).exp().numpy() # log_softmax is faster and has better numerical properties

        a = np.random.choice(self.na, 1, p=a_prob.squeeze())

        return a


    def train(self):
        ids = [worker.work.remote() for worker in self.workers] 
        transition_buffer_list = ray.get(ids)

        for i in range(self.nw):
            self.transition_memory_idx += 1
            if self.transition_memory_idx == self.transition_memory_size:
                self.transition_memory_idx = 0
                self.transition_memory_filled = True

            transition = transition_buffer_list[i]
            
            self.transition_memory[self.transition_memory_idx, :] = transition
            self.transition_buffer[i,:] = transition

        if self.transition_memory_idx > 200:
            self.train_flag = True

        if not self.train_flag:
            return

        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        memory = self.transition_memory[sample_idx,:]
        states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
        actions = memory[:,self.ns]
        rewards = torch.tensor(memory[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor(memory[:,self.ns+2:], dtype=torch.float32)

        self.optim_c.zero_grad()
        Q_target = rewards + self.gamma*self.critic(states_).detach()
        loss_c = F.mse_loss(self.critic(states), Q_target)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 1.0)
        self.optim_c.step()

        
        states = torch.tensor( self.transition_buffer[:,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
        actions = torch.tensor( self.transition_buffer[:,self.ns].astype(int), dtype=torch.int64 ).view(-1)
        rewards = torch.tensor( self.transition_buffer[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor( self.transition_buffer[:,self.ns+2:], dtype=torch.float32 ).view(-1,self.ns)

        self.optim_a.zero_grad()
        advantage = ( rewards + self.gamma*self.critic(states_).detach() - self.critic(states).detach() ).view(-1)      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
        ratio = ( F.log_softmax(self.actor(states), dim=1) - F.log_softmax(self.actor_old(states).detach(), dim=1) ).exp()
        ratio = torch.sum( ratio * F.one_hot(actions, self.na), dim=1)
        loss_a = -torch.min( advantage*ratio, advantage*torch.clamp(ratio, 1-0.2, 1+0.2) ).mean()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), 1.0)
        self.optim_a.step()

        ids = [worker.set_parameters.remote(self.actor, self.critic) for worker in self.workers]
        ray.get(ids)

        if self.transition_memory_idx % 5 == 0:
            self.actor_old.load_state_dict(self.actor.state_dict())


# ------------ DPPO ------------
@ray.remote
class WorkerDPPO():
    def __init__(self, n_states, n_actions):
        self.ns, self.na = n_states, n_actions
        self.actor, self.critic = ActorNetwork(self.ns,2*self.na), CriticNetwork(self.ns)
        self.env = CartpoleEnv_continuous("./models/cartpole.xml")

        self.k_ep, self.k_step = 0, 0
        self.state = self.env.reset()

    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        out = self.actor(x).detach()
        a = torch.normal(mean=out[0,0], std=out[0,1]).numpy()

        return a

    def work(self):
        
        self.k_step += 1

        # self.env.render()

        action = self.action(self.state)
        state_, reward, done = self.env.step(action)

        buffer = np.hstack((self.state,action,reward,state_))

        self.state = state_

        if done:
            print("Ep ",self.k_ep, " step=", self.k_step)
            self.k_ep += 1
            self.k_step = 0
            self.state = self.env.reset()

        return buffer

    def set_parameters(self, actor:ActorNetwork, critic:CriticNetwork):
        self.actor.load_state_dict(actor.state_dict())
        self.critic.load_state_dict(critic.state_dict())
                    


class rlDPPO():
    def __init__(self, n_states, n_actions, n_workers):
        self.ns, self.na, self.nw = n_states, n_actions, n_workers
        self.gamma, self.train_flag = 0.9, False

        self.transition_memory_idx, self.transition_memory_size = -1, 5000
        self.transition_memory_filled = False
        self.batch_size = 64
        self.transition_memory = np.zeros((self.transition_memory_size, self.ns + 1 + 1 + self.ns))
        self.transition_buffer = np.zeros((self.nw, self.ns + 1 + 1 + self.ns))
        
        self.actor = ActorNetwork(in_features=self.ns, out_features=2*self.na)
        self.critic = CriticNetwork(in_features=self.ns)
        self.actor_old = ActorNetwork(in_features=self.ns, out_features=2*self.na)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optim_a = torch.optim.Adam(self.actor.parameters())
        self.optim_c = torch.optim.Adam(self.critic.parameters())

        self.workers = [WorkerDPPO.remote(self.ns, self.na) for _ in range(self.nw)]
        ids = [worker.set_parameters.remote(self.actor, self.critic) for worker in self.workers]
        ray.get(ids)


    def action(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(-1,self.ns)
        out = self.actor(x).detach()
        a = torch.normal(mean=out[0,0], std=out[0,1]).numpy()

        return a


    def train(self):
        ids = [worker.work.remote() for worker in self.workers]
        transition_buffer_list = ray.get(ids)

        for i in range(self.nw):
            self.transition_memory_idx += 1
            if self.transition_memory_idx == self.transition_memory_size:
                self.transition_memory_idx = 0
                self.transition_memory_filled = True

            transition = transition_buffer_list[i]
            
            self.transition_memory[self.transition_memory_idx, :] = transition
            self.transition_buffer[i,:] = transition

        if self.transition_memory_idx > 200:
            self.train_flag = True

        if not self.train_flag:
            return

        if self.transition_memory_filled:
            sample_idx = np.random.choice(self.transition_memory_size, self.batch_size)
        else:
            sample_idx = np.random.choice(self.transition_memory_idx, self.batch_size)

        memory = self.transition_memory[sample_idx,:]
        states = torch.tensor(memory[:,:self.ns], dtype=torch.float32)
        actions = memory[:,self.ns]
        rewards = torch.tensor(memory[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor(memory[:,self.ns+2:], dtype=torch.float32)

        self.optim_c.zero_grad()
        Q_target = rewards + self.gamma*self.critic(states_).detach()
        loss_c = F.mse_loss(self.critic(states), Q_target)
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 1.0)
        self.optim_c.step()

        
        states = torch.tensor( self.transition_buffer[:,:self.ns], dtype=torch.float32 ).view(-1,self.ns)
        actions = torch.tensor( self.transition_buffer[:,self.ns], dtype=torch.float32 ).view(-1)
        rewards = torch.tensor( self.transition_buffer[:,self.ns+1], dtype=torch.float32).view(-1,1)
        states_ = torch.tensor( self.transition_buffer[:,self.ns+2:], dtype=torch.float32 ).view(-1,self.ns)

        self.optim_a.zero_grad()
        advantage = ( rewards + self.gamma*self.critic(states_).detach() - self.critic(states).detach() ).view(-1)      # A(s,a) = Q(s,a) - V(s), where Q(s,a) = r + gamma*V(s')
        mu_sigma, mu_sigma_old = self.actor(states), self.actor_old(states).detach()
        torch.clamp_min_(mu_sigma[:,1], 0.05)
        torch.clamp_min_(mu_sigma_old[:,1], 0.05)
        dist, dist_old = torch.distributions.Normal(mu_sigma[:,0], mu_sigma[:,1]), torch.distributions.Normal(mu_sigma_old[:,0], mu_sigma_old[:,1])
        ratio = ( dist.log_prob(actions) - dist_old.log_prob(actions) ).exp()
        loss_a = -torch.min( advantage*ratio, advantage*torch.clamp(ratio, 1-0.2, 1+0.2) ).mean()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), 1.0)
        self.optim_a.step()

        ids = [worker.set_parameters.remote(self.actor, self.critic) for worker in self.workers]
        ray.get(ids)

        if self.transition_memory_idx % 5 == 0:
            self.actor_old.load_state_dict(self.actor.state_dict())



if __name__ == "__main__":
    # ray.init()
    
    # # agent = rlA3C(4,2,10)

    # agent = rlA2C(4,2)
    # workers = [WorkerPPO.remote(agent_global=agent, n_states=4, n_actions=2) for _ in range(5)]

    # ids = [worker.work.remote() for worker in workers]
    
    # ray.get(ids)

    # ray.shutdown()

    # # agent = rlA2C(4,2)
    # # worker = WorkerPPO(agent,4,2)
    # # worker.work()


    # ------------ DPPO ------------
    ray.init(num_cpus=6)

    agent = rlDPPO(4,1,6)

    for k_step in range(100000):
        agent.train()

    ray.shutdown()