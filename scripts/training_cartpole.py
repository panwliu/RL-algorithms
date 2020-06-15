import numpy as np
import environments

import matplotlib.pyplot as plt

env = environments.CartpoleEnv(port_self=18060, port_remote=18080)

n_ep = 3
n_step = 1000
state_all = np.zeros((7,n_step))
force_cmd_all = np.zeros(n_step)
for k_ep in range(n_ep):
    state = env.reset(model_id=0)

    for k_step in range(n_step):
        
        if k_step<500:
            force_cmd = -0.5
        else:
            force_cmd = 0.5

        env.sendCommands(model_id=0, cmd_type="force_cmd", command=force_cmd)
        state = env.step()
        state_all[:,k_step] = env.state_current_[:]
        force_cmd_all[k_step] = force_cmd

time = state_all[2,:]
pos = state_all[3,:]
angle = state_all[4,:]
vel = state_all[5,:]
omega = state_all[6,:]
plt.figure()
plt.subplot(2,2,1)
l1, = plt.plot(time[::20], pos[::20])
plt.legend([l1], ["position"], loc="upper right")
plt.xlabel("Time (ms)")
plt.ylabel("Position (m)")
plt.title("Position")

# plt.subplot(2,2,1)
# l1, = plt.plot(time[::20], force_cmd_all[0, ::20])
# plt.legend([l1], ["force cmd"], loc="upper right")
# plt.xlabel("Time (ms)")
# plt.ylabel("Force (N)")
# plt.title("Force")

plt.subplot(2,2,2)
l1, = plt.plot(time[::20], vel[::20])
plt.legend([l1], ["velocity"], loc="upper right")
plt.xlabel("Time (ms)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity")

plt.subplot(2,2,3)
l1, = plt.plot(time[::20], angle[::20]*180/np.pi)
plt.legend([l1], ["angle"], loc="upper right")
plt.xlabel("Time (ms)")
plt.ylabel("Angle (deg) ")
plt.title("Angle")

plt.subplot(2,2,4)
l1, = plt.plot(time[::20], omega[::20]*180/np.pi)
plt.legend([l1], ["rate"], loc="upper right")
plt.xlabel("Time (ms)")
plt.ylabel("Angular Velocity (deg/s)")
plt.title("Angular velocity")

plt.show()


