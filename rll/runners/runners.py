import numpy as np
import torch
import rll
import time

def reset_parameters(m):        # default reset fcn of torch.nn.Linear  https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(m.bias, -bound, bound)
        

def onpolicy_runner(env:rll.envs.EnvBase, agent: rll.algos.rlBase, buffer: rll.buffers.BufferBase, logger: rll.utils.loggers.LoggerBase, args):
    epochs = args.epochs
    sample_size = args.sample_size_local
    num_procs = args.num_procs

    
    rll.utils.mpi_tools.mpi_fork(num_procs, host=args.mpi_host)
    proc_id = rll.utils.mpi_tools.proc_id()

    rll.utils.mpi_tools.setup_pytorch_for_mpi()

    seed = 0
    seed += 10000 * proc_id
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent.actor.apply(reset_parameters)
    rll.utils.mpi_tools.sync_params(agent.actor)
    if agent.critic:
        agent.critic.apply(reset_parameters)
        rll.utils.mpi_tools.sync_params(agent.critic)
    

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
                buffer.finish_traj(critic=agent.critic)
                obs = env.reset()
        
        if not done:
            k_ep += 1
            last_val = agent.critic(torch.tensor(obs,dtype=torch.float32)).detach().numpy() if agent.critic else 0
            buffer.finish_traj(critic=agent.critic, last_val=last_val)
        
        sampling_time = time.time() - epoch_start_time
        agent.train(buffer)
        training_time = time.time() - epoch_start_time - sampling_time

        if proc_id == 0:
            print( 'Epoch: %3d \t return: %.3f \t ep_len: %.3f' %(k_epoch, np.mean(buffer.reward_to_go_buf), sample_size/k_ep), flush=True )
            print('Tsp: %.3f \t Ttr: %.3f' %(sampling_time, training_time), flush=True)
            logger.write_reward(k_epoch, np.mean(buffer.reward_buf), np.mean(buffer.reward_to_go_buf), sample_size/k_ep, sampling_time, training_time)
            if (k_epoch+1) % args.save_freq == 0:
                logger.save_checkpoint(filename=args.env+'-Epoch'+str(k_epoch)+'-proc' + str(proc_id)+'.tar', agent=agent, args=args)




