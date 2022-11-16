import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation  = env.reset(seed=42)

def policy(observation):
    return env.action_space.sample()

for _ in range(1000):
    action = policy(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()


