import numpy as np
import torch
import rll
import argparse

parser = argparse.ArgumentParser()
# ------------ CartPole ------------
parser.add_argument('--env', type=str, default='CartPole-v1')
parser.add_argument('--hid', type=list, default=[32])
parser.add_argument('--act_std', type=float, default=0.2)
parser.add_argument('--lr_a', type=float, default=1e-3)
parser.add_argument('--lr_c', type=float, default=1e-3)
parser.add_argument('--train_a_itrs', type=int, default=60)
parser.add_argument('--train_c_itrs', type=int, default=60)
parser.add_argument('--target_kl', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--sample_size', type=int, default=5000)
parser.add_argument('--num_procs', type=int, default=4)
parser.add_argument('--save_freq', type=int, default=100)
# # ------------ Walker2D ------------
# parser.add_argument('--env', type=str, default='Walker2D-v1')
# parser.add_argument('--hid', type=list, default=[64,64])
# parser.add_argument('--act_std', type=float, default=0.2)
# parser.add_argument('--lr_a', type=float, default=3e-4)
# parser.add_argument('--lr_c', type=float, default=1e-3)
# parser.add_argument('--train_a_itrs', type=int, default=80)
# parser.add_argument('--train_c_itrs', type=int, default=80)
# parser.add_argument('--target_kl', type=float, default=0.05)
# parser.add_argument('--epochs', type=int, default=5000)
# parser.add_argument('--sample_size', type=int, default=3000)
# parser.add_argument('--num_procs', type=int, default=8)
# parser.add_argument('--save_freq', type=int, default=100)

args = parser.parse_args()


## hyperparameters
env_name = args.env
hid_size = args.hid
act_std = args.act_std
lr_a = args.lr_a
lr_c = args.lr_c
args.sample_size_local = args.sample_size//args.num_procs


## create env, agent and buffer
env = rll.envs.env_by_name(env_name)

actor_std = act_std*np.ones(env.act_dim, dtype=np.float32)
actor = rll.utils.nns.MLPGaussianActor(env.obs_dim, env.act_dim, hid_size, torch.nn.ReLU, actor_std)
critic = rll.utils.nns.MLPCritic(env.obs_dim, hid_size, torch.nn.ReLU)
optimizer_a = torch.optim.Adam(actor.parameters(), lr=lr_a)
optimizer_c = torch.optim.Adam(critic.parameters(), lr=lr_c)
agent = rll.algos.rlPPO(actor=actor, critic=critic, optimizers=[optimizer_a, optimizer_c], args=args)

buffer = rll.buffers.ExpBuffer(env.obs_dim, env.act_dim, args.sample_size_local)

log_dir = './log/' + env_name + '-PPO'
logger = rll.utils.loggers.Logger(log_dir)


## run training
rll.runners.onpolicy_runner(env, agent, buffer, logger, args)