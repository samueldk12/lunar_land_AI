import gym
import os
import pygame
import platform

try:
    os.environ["DISPLAY"]
    print('run in dysplay mode')
except:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    print('run in dummy mode')

drivers = ['fbcon', 'directfb', 'svgalib','directx','x11','dga','ggi','aalib','wayland','kmsdrm']
found = False
for driver in drivers:
    # Make sure that SDL_VIDEODRIVER is set
    if not os.getenv('SDL_VIDEODRIVER'):
        os.putenv('SDL_VIDEODRIVER', driver)
    try:
        pygame.display.init()
    except pygame.error:
        print('Driver: {0} failed.'.format(driver))
        continue
    found = True
    break

if not found:
    #raise Exception('No suitable video driver found!')
    print('no video driver found')
env = gym.make("LunarLander-v2", render_mode="human")

env.reset() # Instantiate enviroment with default parameters
for step in range(300):
    env.render() # Show agent actions on screen
    env.step(env.action_space.sample()) # Sample random action
    print('step: ', step)
env.close()
