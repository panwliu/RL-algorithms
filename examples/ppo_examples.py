import numpy as np
import torch
import rll
import argparse

parser = argparse.ArgumentParser()
# ------------ CartPole ------------
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--hid', type=list, default=[32])
parser.add_argument('--act_std', type=float, default=0.2)
parser.add_argument('--lr_a', type=float, default=1e-2)
parser.add_argument('--lr_c', type=float, default=1e-2)
parser.add_argument('--train_a_itrs', type=int, default=20)
parser.add_argument('--train_c_itrs', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--sample_size', type=int, default=5000)
parser.add_argument('--num_procs', type=int, default=4)
# # ------------ Walker2D ------------
# parser.add_argument('--env', type=str, default='Walker2D-v1')
# parser.add_argument('--hid', type=list, default=[64,64])
# parser.add_argument('--act_std', type=float, default=0.2)
# parser.add_argument('--lr_a', type=float, default=1e-3)
# parser.add_argument('--lr_c', type=float, default=1e-2)
# parser.add_argument('--train_a_itrs', type=int, default=30)
# parser.add_argument('--train_c_itrs', type=int, default=5)
# parser.add_argument('--epochs', type=int, default=5000)
# parser.add_argument('--sample_size', type=int, default=3000)
# parser.add_argument('--num_procs', type=int, default=1)

args = parser.parse_args()


## hyperparameters
env_name = args.env
hid_size = args.hid
act_std = args.act_std
lr_a = args.lr_a
lr_c = args.lr_c
train_itrs = [args.train_a_itrs, args.train_c_itrs]
epochs = args.epochs
sample_size = args.sample_size // args.num_procs
num_procs = args.num_procs


## create env, agent and buffer
env = rll.envs.env_by_name(env_name)

actor_std = act_std*np.ones(env.act_dim, dtype=np.float32)
actor = rll.utils.nns.MLPGaussianActor(env.obs_dim, env.act_dim, hid_size, torch.nn.ReLU, actor_std)
critic = rll.utils.nns.MLPCritic(env.obs_dim, hid_size, torch.nn.ReLU)
optimizer_a = torch.optim.Adam(actor.parameters(), lr=lr_a)
optimizer_c = torch.optim.Adam(critic.parameters(), lr=lr_c)
agent = rll.algos.rlPPO(actor=actor, critic=critic, optimizers=[optimizer_a, optimizer_c], train_itrs=train_itrs)

buffer = rll.buffers.ExpBuffer(env.obs_dim, env.act_dim, sample_size)


## run training
param = {'epochs': epochs, 'sample_size': sample_size, 'num_procs': num_procs}
rll.runners.onpolicy_runner(env, agent, buffer, param)