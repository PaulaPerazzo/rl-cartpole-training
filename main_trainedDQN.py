import gymnasium as gym
from stable_baselines3 import DQN

env = gym.make('CartPole-v1', render_mode='human')

model = DQN('MlpPolicy', env, verbose=1).learn(10000)

vec_env = model.get_env()
obs = vec_env.reset()

count = 0
for _ in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    
    count += 1

    if done:
        print(info)
        obs = vec_env.reset()

env.close()
