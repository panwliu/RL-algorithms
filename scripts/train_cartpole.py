import numpy as np
import mujocosim

sim = mujocosim.MujocoSim("../models/cartpole.xml")
viewer = mujocosim.MujocoVis(sim)

while not viewer.window_should_close():
    action = -np.random.rand(1)
    state = sim.step(action)
    viewer.render()
