import gymenv = gym.make("LunarLander-v2")
env.reset() # Instantiate enviroment with default parameters
for step in range(300):
    env.render() # Show agent actions on screen
    env.step(env.action_space.sample()) # Sample random action
env.close()