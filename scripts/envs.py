import numpy as np
import mujocosim

class CartpoleEnv:
    def __init__(self, model_path="../models/cartpole.xml"):
        self.sim = mujocosim.MujocoSim(model_path)
        self.viewer = mujocosim.MujocoVis(self.sim)

    def step(self, action):
        if action == 0:
            action = -1.0
        elif action == 1:
            action = 1.0
        else:
            print("Unknown action!")

        state = self.sim.step(action)

        x, theta, x_dot, theta_dot = state
        r1 = (0.7 - np.abs(x))/0.7 - 0.5
        r2 = (0.7 - np.abs(theta))/0.7 - 0.5
        reward = r1 + r2
        # reward  = 1

        if np.abs(state[0])>0.7 or np.abs(state[1])>0.7 or self.viewer.window_should_close():
            done = True
            reward = -10
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
        self.viewer.render()
        
class CartpoleEnv_continuous:
    def __init__(self, model_path="../models/cartpole.xml"):
        self.sim = mujocosim.MujocoSim(model_path)
        self.viewer = mujocosim.MujocoVis(self.sim)

    def step(self, action):
        action = np.clip(action, -2.0, 2.0)

        state = self.sim.step(action)

        x, theta, x_dot, theta_dot = state
        r1 = (0.7 - np.abs(x))/0.7 - 0.5
        r2 = (0.7 - np.abs(theta))/0.7 - 0.5
        reward = r1 + r2
        # reward  = 1

        if np.abs(state[0])>0.7 or np.abs(state[1])>0.7 or self.viewer.window_should_close():
            done = True
            reward = -10
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
        self.viewer.render()
        

if __name__=="__main__":
    env = CartpoleEnv()

    for _ in range(10):
        done = False

        s = env.reset()
        while not done:
            env.render()

            a = np.random.choice(2,1)

            s_, r, done = env.step(a)

            s = s_


if __name__=="__main__":
    env = CartpoleEnv()

    for _ in range(10):
        done = False

        s = env.reset()
        while not done:
            env.render()

            a = np.random.choice(2,1)

            s_, r, done = env.step(a)

            s = s_