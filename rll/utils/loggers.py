from torch.utils.tensorboard import SummaryWriter
import rll
import time

class Logger:
    def __init__(self, log_dir) -> None:
        self.proc_id = rll.utils.mpi_tools.proc_id()
        self.log_dir = log_dir + time.strftime( '-%Y-%m-%d_%H-%M-%S', time.localtime(time.time()) )
        if self.proc_id == 0:
            self.writer = SummaryWriter(log_dir = self.log_dir)

    def write_reward(self, k_epoch, reward, r2g, ep_len, sampling_time, training_time):
        self.writer.add_scalar("Results/Reward", reward, k_epoch)
        self.writer.add_scalar("Results/R2G", r2g, k_epoch)
        self.writer.add_scalar("Results/Ep_len", ep_len, k_epoch)
        self.writer.add_scalar("Info/Tsp", sampling_time, k_epoch)
        self.writer.add_scalar("Info/Ttr", training_time, k_epoch)