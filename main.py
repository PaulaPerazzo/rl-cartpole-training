import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(reward)

        # tirar isso aqui pra testar
        # obs, info = env.reset()

env.close()
