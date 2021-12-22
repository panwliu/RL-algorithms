import numpy as np
import torch
import rll
import time

def onpolicy_runner(env:rll.envs.EnvBase, agent: rll.algos.rlBase, buffer: rll.buffers.BufferBase, param_dict: dict):
    epochs = param_dict['epochs']
    sample_size = param_dict['sample_size']
    num_procs = param_dict['num_procs']

    for k_epoch in range(epochs):
        epoch_start_time = time.time()
        obs = env.reset()

        k_ep = 0
        for k_sample in range(sample_size):

            act, logp = agent.action(obs)

            obs_, reward, done, _ = env.step(act)

            buffer.store(obs, act, reward, obs_, logp)

            obs = obs_

            if done:
                k_ep += 1
                buffer.finish_traj()
                obs = env.reset()
        
        if not done:
            k_ep += 1
            last_val = agent.critic(torch.tensor(obs,dtype=torch.float32)).detach().numpy() if agent.critic else 0
            buffer.finish_traj(last_val=last_val)
        
        sampling_time = time.time() - epoch_start_time
        agent.train(buffer)
        training_time = time.time() - epoch_start_time - sampling_time

        print( 'Epoch: %3d \t return: %.3f \t ep_len: %.3f' %(k_epoch, torch.mean(buffer.reward_to_go_buf), sample_size/k_ep), flush=True )
        print('Tsp: %.3f \t Ttr: %.3f' %(sampling_time, training_time), flush=True)




