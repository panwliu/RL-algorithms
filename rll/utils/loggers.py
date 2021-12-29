import torch
from torch.utils.tensorboard import SummaryWriter
import rll
import time

class LoggerBase:
    def write_reward(self):
        raise NotImplementedError
    def save_checkpoint(self):
        raise NotImplementedError

class Logger:
    def __init__(self, log_dir) -> None:
        self.proc_id = rll.utils.mpi_tools.proc_id()
        self.log_dir = log_dir + time.strftime( '-%Y-%m-%d_%H-%M-%S', time.localtime(time.time()) )
        self.writer = None

    def write_reward(self, k_epoch, reward, r2g, ep_len, sampling_time, training_time):
        if not self.writer:
            self.writer = SummaryWriter(log_dir = self.log_dir)
        self.writer.add_scalar("Results/Reward", reward, k_epoch)
        self.writer.add_scalar("Results/R2G", r2g, k_epoch)
        self.writer.add_scalar("Results/Ep_len", ep_len, k_epoch)
        self.writer.add_scalar("Info/Tsp", sampling_time, k_epoch)
        self.writer.add_scalar("Info/Ttr", training_time, k_epoch)

    def save_checkpoint(self, filename, agent, args):
        path = self.log_dir + '/' + filename
        torch.save( {'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer_state_dict': agent.optimizer_a.state_dict(),
                    'critic_optimizer_state_dict': agent.optimizer_c.state_dict(),
                    'args': args},
                    path )