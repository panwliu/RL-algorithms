import numpy as np
import torch
import rll
import argparse

parser = argparse.ArgumentParser()
# # ------------ CartPole ------------
# parser.add_argument('--env', type=str, default='CartPole-v1')
# parser.add_argument('--hid', type=list, default=[32])
# parser.add_argument('--act_std', type=float, default=0.2)
# parser.add_argument('--lr_a', type=float, default=1e-3)
# parser.add_argument('--lr_c', type=float, default=1e-3)
# parser.add_argument('--train_a_itrs', type=int, default=60)
# parser.add_argument('--train_c_itrs', type=int, default=60)
# parser.add_argument('--target_kl', type=float, default=0.01)
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--sample_size', type=int, default=5000)
# parser.add_argument('--num_procs', type=int, default=4)
# parser.add_argument('--mpi_host', type=str, default='')
# parser.add_argument('--save_freq', type=int, default=100)
# ------------ Walker2D ------------
parser.add_argument('--env', type=str, default='Walker2D-v1')
parser.add_argument('--hid', type=list, default=[64,64])
parser.add_argument('--act_std', type=float, default=0.2)
parser.add_argument('--lr_a', type=float, default=3e-4)
parser.add_argument('--lr_c', type=float, default=1e-3)
parser.add_argument('--train_a_itrs', type=int, default=2**4)
parser.add_argument('--train_c_itrs', type=int, default=2**4)
parser.add_argument('--batch_size', type=int, default=2**9)
parser.add_argument('--target_kl', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--sample_size', type=int, default=2**12)
parser.add_argument('--max_ep_length', type=int, default=2**8)
parser.add_argument('--norm_epochs', type=int, default=0)
parser.add_argument('--num_procs', type=int, default=8)
parser.add_argument('--mpi_host', type=str, default='')
parser.add_argument('--save_freq', type=int, default=100)

args = parser.parse_args()


## hyperparameters
env_name = args.env
hid_size = args.hid
act_std = args.act_std
lr_a = args.lr_a
lr_c = args.lr_c
args.batch_size_local = int( args.batch_size / args.num_procs )
args.sample_size_local = int( args.sample_size / args.num_procs )
args.obs_mean = np.array([1.1859156, -0.48713082, -0.3154162, -0.36796355, 0.23148866, -0.3162046, -0.36756027, 0.22122139, -1.2135289, -1.8748417,
                            -12.068233, -9.955445, -5.3886685, 1.7908473, -9.967251, -5.3825417, 1.699529])
args.obs_std = np.array([0.09646927, 0.44239914, 0.46256107, 0.5386492, 0.5400577, 0.4629824, 0.53765464, 0.5460499, 1.0368233, 2.2401385, 13.779417,
                            17.85323, 10.55298, 19.291592, 17.860907, 10.536373, 19.319115])


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