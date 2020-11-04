import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import envs
import ray

# ------------ DQN ------------ 

            

# ------------ PG ------------ 


# ------------ A2C ------------ 




# ------------ A3C ------------



# ------------ PPO ------------






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