import numpy as np
from envs import CartpoleEnv
from rl import rlDQN, rlPG, rlA2C

# ------------ DQN/A2C ------------ 
env = CartpoleEnv("./models/cartpole.xml")
# agent = rlDQN()
agent = rlA2C(n_states=4, n_actions=2)

for k_ep in range(1000):
    state = env.reset()

    k_step = 0
    while True:
        k_step += 1

        # env.render()

        action = agent.action(state)
        state_, reward, done = env.step(action)

        agent.store_transition(state, action, reward, state_)

        if k_ep > 10:
            agent.train()

        state = state_

        if done:
            print("Ep ",k_ep, " step=", k_step)
            break


# # ------------ PG ------------ 
# env = CartpoleEnv("./models/cartpole.xml")
# agent = rlPG(n_states=4, n_actions=2)

# for k_ep in range(1000):
#     state = env.reset()

#     k_step = 0
#     while True:
#         k_step += 1

#         # env.render()

#         action = agent.action(state)
#         state_, reward, done = env.step(action)

#         agent.store_transition(state, action, reward)

#         state = state_

#         if done:
#             print("Ep ",k_ep, " step=", k_step)
#             agent.train()
#             break

