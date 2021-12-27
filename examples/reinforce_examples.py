import numpy as np
import torch
import rll
import argparse

parser = argparse.ArgumentParser()
# ------------ CartPole ------------
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--hid', type=list, default=[32,32])
parser.add_argument('--act_std', type=float, default=0.6)         # sigma > 0.4
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--num_procs', type=int, default=4)
# # ------------ Walker2D ------------
# parser.add_argument('--env', type=str, default='Walker2D-v1')
# parser.add_argument('--hid', type=list, default=[128,128])
# parser.add_argument('--act_std', type=float, default=0.6)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--epochs', type=int, default=1000)
# parser.add_argument('--sample_size', type=int, default=80000)
# parser.add_argument('--num_procs', type=int, default=8)

args = parser.parse_args()


## hyperparameters
env_name = args.env
hid_size = args.hid
act_std = args.act_std
lr = args.lr
epochs = args.epochs
sample_size = args.sample_size // args.num_procs
num_procs = args.num_procs


## create env, agent and buffer
env = rll.envs.env_by_name(env_name)

actor_std = act_std*np.ones(env.act_dim, dtype=np.float32)
actor = rll.utils.nns.MLPGaussianActor(env.obs_dim, env.act_dim, hid_size, torch.nn.ReLU, actor_std)
optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
agent = rll.algos.rlReinforce(actor=actor, optimizer=optimizer)

buffer = rll.buffers.ExpBuffer(env.obs_dim, env.act_dim, sample_size)


## run training
param = {'epochs': epochs, 'sample_size': sample_size, 'num_procs': num_procs}
rll.runners.onpolicy_runner(env, agent, buffer, param)