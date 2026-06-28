import gymnasium as gym

env = gym.make("Ant-v4", render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, info = env.reset()