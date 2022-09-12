import gym

env = gym.make('SpaceInvaders-v0', render_mode='human')

env.reset()

episode = 0
for _ in range(1000):

    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)

    if done:
        episode += 1
    if episode >= 10:
        break