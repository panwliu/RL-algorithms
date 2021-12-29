import numpy as np
import torch
import rll
import time
import argparse

def eval(path, play_speed=1):
        checkpoint = torch.load(path)
        args = checkpoint['args']

        env = rll.envs.env_by_name(args.env)
        actor = rll.utils.nns.MLPGaussianActor(env.obs_dim, env.act_dim, args.hid, torch.nn.ReLU, 0.0001)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()

        sim_timestep, simrate = env.sim_timestep, env.simrate
        render_rate = 0.03 // (sim_timestep*simrate)
        render_rate = np.max((render_rate,1))
        
        for _ in range(5):
            obs = env.reset()
            for j in range(300):
                sim_starttime = time.time()
                act, _ = actor.action(obs)
                obs, _, done, _ = env.step(act)
                if (j%render_rate) == 0:
                    env.render()
                    while time.time() - sim_starttime < sim_timestep*simrate / play_speed:
                        time.sleep(sim_timestep/play_speed)
                if done:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='')
    args = parser.parse_args()

    eval(args.log_dir)