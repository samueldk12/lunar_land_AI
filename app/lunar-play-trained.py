import gym
import numpy as np
import tensorflow as tf

env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -11.0,
    enable_wind = True,
    wind_power = 15.0,
    turbulence_power = 1.5,
)

q_network = tf.keras.models.load_model('./models/medium.h5')


def run(env, q_network):
    done = False
    state = env.reset()
    while not done:
        state = np.expand_dims(state, axis=0)
        q_values = q_network(state)
        action = np.argmax(q_values.numpy()[0])
        state, _, done, _ = env.step(action)
        env.render()


run(env, q_network)