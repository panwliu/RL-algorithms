import numpy as np
import mujocosim
import os, pathlib, time
import torch, nns

def env_by_name(env_name):
    file_path = pathlib.Path( os.path.realpath(__file__) )
    models_path = file_path.parent.parent.__str__() + '/models/'

    if env_name == 'CartPole-v0':
        model_file = 'cartpole.xml'
        env = CartpoleEnv(model_path = models_path + model_file)
    elif env_name == 'CartPole-v1':     # continuous
        model_file = 'cartpole.xml'
        env = CartpoleEnv(model_path = models_path + model_file, discrete=False)
    elif env_name == 'Walker2D-v0':
        model_file = 'walker2d.xml'
        env = Walker2dEnv(model_path = models_path + model_file)
    elif env_name == 'Walker2D-v1':     # jumping
        model_file = 'walker2d.xml'
        env = Walker2dEnv(model_path = models_path + model_file, jumping=True)
    elif env_name == 'Cassie-v0':
        model_file = 'cassie.xml'
        env = CassieEnv(model_path = models_path + model_file)
    else:
        print("Can't find env " + env_name)
    
    return env


class CartpoleEnv:
    def __init__(self, model_path="../models/cartpole.xml", discrete=True):
        self.sim = mujocosim.MujocoSim(model_path)
        self.sim_timestep, self.simrate = 0.005, 4
        self.discrete = discrete
        if discrete:
            self.obs_dim, self.act_dim = 4, 2
        else:
            self.obs_dim, self.act_dim = 4, 1

    def step(self, action):
        if self.discrete:
            if action == 0:
                action = -1.0
            elif action == 1:
                action = 1.0
            else:
                print("Unknown action!")
        else:
            action = np.clip(action, -1.0, 1.0)

        for _ in range(self.simrate):
            state = self.sim.step(action)

        # x, theta, x_dot, theta_dot = state
        # r1 = (0.7 - np.abs(x))/0.7 - 0.5
        # r2 = (0.7 - np.abs(theta))/0.7 - 0.5
        # reward = r1 + r2
        reward  = 1

        if np.abs(state[0])>0.7 or np.abs(state[1])>0.7:
            done = True
            # reward = -10
        else:
            done = False

        return state, reward, done

    def reset(self):
        pose = np.random.randn(2)*0.1
        pose = np.clip(pose, -0.2, 0.2)
        pose[0]=0

        state = self.sim.reset(pose)        # state = sim.reset([])
        return state

    def render(self):
        if not hasattr(self, "viewer"):
            self.viewer = mujocosim.MujocoVis(self.sim)
        self.viewer.render()
        
    def eval(self, path, play_speed=1):
        checkpoint = torch.load(path)
        args = checkpoint['args']
        actor = nns.MLPGaussianActor(self.obs_dim, self.act_dim, args.hid, torch.nn.ReLU, 0.3)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        
        for _ in range(5):
            obs = self.reset()
            for _ in range(1000):
                self.render()
                act, _ = actor.action(obs)
                sim_starttime = time.time()
                obs, _, _ = self.step(act)
                while time.time() - sim_starttime < self.sim_timestep*self.simrate / play_speed:
                    time.sleep(self.sim_timestep/play_speed)
        
class Walker2dEnv:
    def __init__(self, model_path="../models/walker2d.xml", jumping=False):
        self.sim = mujocosim.MujocoSim(model_path)
        self.sim_timestep, self.simrate = 0.002, 10
        self.jumping = jumping
        if jumping:
            self.obs_dim, self.act_dim = 17, 3
        else:
            self.obs_dim, self.act_dim = 17, 6

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        if self.jumping:
            action = np.hstack((action,action))
        for _ in range(self.simrate):
            state = self.sim.step(action)

        height, angle, vel = state[1], state[2], state[9]
        
        reward  = 1 + vel

        done = not (height > 0.8 and height < 2.0 and angle > -1.0 and angle < 1.0)

        return state[1:], reward, done

    def reset(self):
        state = self.sim.reset([])
        return state[1:]

    def render(self):
        if not hasattr(self, "viewer"):
            self.viewer = mujocosim.MujocoVis(self.sim)
        self.viewer.render()

    def eval(self, path, play_speed=1):
        checkpoint = torch.load(path)
        args = checkpoint['args']
        actor = nns.MLPGaussianActor(self.obs_dim, self.act_dim, args.hid, torch.nn.ReLU, 0.2)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        
        for _ in range(5):
            obs = self.reset()
            for _ in range(500):
                self.render()
                act, _ = actor.action(obs)
                sim_starttime = time.time()
                obs, _, done = self.step(act)
                while time.time() - sim_starttime < self.sim_timestep*self.simrate / play_speed:
                    time.sleep(self.sim_timestep/play_speed)
                if done:
                    break



class CassieEnv:
    def __init__(self, model_path="../models/cassie.xml", reward_type='standing'):
        self.sim = mujocosim.MujocoSim(model_path)
        self.sim_timestep, self.simrate = 0.0005, 50
        self.obs_dim, self.act_dim = self.cassie_state_fcn(np.zeros(67)).size, 10
        self.reward_type = reward_type

        self.qpos_init = [0, 0, 1.01, 1, 0, 0, 0,
        0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
        -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968]
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.p_gain = np.array([100,  100,  88,  96,  50,    100,  100,  88,  96,  50])
        self.d_gain = np.array([10.0, 10.0, 8.0, 9.6, 5.0,   10.0, 10.0, 8.0, 9.6, 5.0])

    def step(self, action):
        # motor_pos = self.cassie_state[7:17]
        # motor_vel = self.cassie_state[17:27]
        # target = action + self.offset

        # action = self.p_gain*(target - motor_pos) + self.d_gain*(0 - motor_vel)

        action_scale = np.array( [4.5, 4.5, 12.2, 12.2, 0.9, 4.5, 4.5, 12.2, 12.2, 0.9] )
        action = action_scale * action

        for _ in range(self.simrate):
            state = self.sim.step(action)
        self.cassie_state = self.cassie_state_fcn(state)

        height, vel = state[2], state[35]
        
        if self.reward_type == 'standing':
            motor_pos = self.cassie_state[7:17]
            unactuated_joint_pos = self.cassie_state[27:31]
            left_joint_pos = np.hstack((motor_pos[:5], unactuated_joint_pos[:2]))
            right_joint_pos = np.hstack((motor_pos[5:], unactuated_joint_pos[2:]))
            joint_error = np.mean( (left_joint_pos-right_joint_pos)**2 )
            reward = 1 - joint_error
        elif self.reward_type == 'forward':
            reward = 1 + vel
        else:
            print('No reward type ' + self.reward_type)

        done = not (height > 0.7 and height < 2.0 )

        return self.cassie_state, reward, done

    def reset(self):
        state = self.sim.reset(self.qpos_init)
        self.cassie_state = self.cassie_state_fcn(state)
        
        return self.cassie_state

    def render(self):
        if not hasattr(self, "viewer"):
            self.viewer = mujocosim.MujocoVis(self.sim)
        self.viewer.render()

    def cassie_state_fcn(self,state):
        q_pos, q_vel = state[:35], state[35:]
        pelvis_orientation = q_pos[3:7]         # size 4
        pelvis_angularVel = q_vel[3:6]          # size 3
        motor_pos = [q_pos[7], q_pos[8], q_pos[9], q_pos[14], q_pos[20],   q_pos[21], q_pos[22], q_pos[23], q_pos[28], q_pos[34] ]  # size 10
        motor_vel = [q_vel[6], q_vel[7], q_vel[8], q_vel[12], q_vel[18],   q_vel[19], q_vel[20], q_vel[21], q_vel[25], q_vel[31] ]  # size 10
        joint_pos = [q_pos[15], q_pos[16], q_pos[29], q_pos[30] ]   # size 4
        joint_vel = [q_pos[13], q_pos[14], q_pos[26], q_pos[27] ]   # size 4

        cassie_state = np.hstack((pelvis_orientation, pelvis_angularVel, motor_pos, motor_vel, joint_pos, joint_vel, q_pos[2]))

        return cassie_state
    
    def eval(self, path, play_speed=1):
        checkpoint = torch.load(path)
        args = checkpoint['args']
        actor = nns.MLPGaussianActor(self.obs_dim, self.act_dim, args.hid, torch.nn.ReLU, 0.3)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        
        for _ in range(10):
            obs = self.reset()
            for _ in range(500):
                self.render()
                act, _ = actor.action(obs)
                sim_starttime = time.time()
                obs, _, done = self.step(act)
                while time.time() - sim_starttime < self.sim_timestep*self.simrate / play_speed:
                    time.sleep(self.sim_timestep/play_speed)
                if done:
                    break


if __name__=="__main__":
    # env = env_by_name(env_name="CartPole-v0")
    # env = env_by_name(env_name="Walker2D-v0")
    env = env_by_name(env_name="Cassie-v0")

    for _ in range(10):
        done = False

        s = env.reset()
        while not done:
            env.render()

            a = np.zeros(env.act_dim)
            # a[0] = 1
            # a[3] = -1

            s_, r, done = env.step(a)

            s = s_

