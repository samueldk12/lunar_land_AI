import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np

# Game
env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -11.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
)

env.action_space.seed(0)
np.random.seed(0)


def build_model():
    model = Sequential()
    model.add(Dense(150, input_dim=state_space, activation=relu))
    model.add(Dense(120, activation=relu))
    model.add(Dense(action_space, activation=linear))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


# Deep q learning
global action_space
action_space = None
global state_space
state_space = None
global epsilon
epsilon = 1.0
gamma = .99
batch_size = 64
epsilon_min = .01
learning_rate = 0.001
epsilon_decay = .996
memory = deque(maxlen=1000000)
model = Sequential()


#env.reset() # Instantiate enviroment with default parameters
#for step in range(300):
    #env.render() # Show agent actions on screen
    #env.step(env.action_space.sample()) # Sample random action
    #time.sleep(1) descoment to see in 'slow'
#env.close()


def train_dqn(episode):
    loss = []

    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 500
        for i in range(max_steps):

            action = act(state)
            env.render()

            next_state, reward, done, _ = env.step(action)
            score += reward

            next_state = np.reshape(next_state, (1, 8))
            remember(state, action, reward, next_state, done)
            state = next_state
            replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_space)
    act_values = model.predict(state)
    return np.argmax(act_values[0])


def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states = np.array([i[0] for i in minibatch])
    actions = np.array([i[1] for i in minibatch])
    rewards = np.array([i[2] for i in minibatch])
    next_states = np.array([i[3] for i in minibatch])
    dones = np.array([i[4] for i in minibatch])

    states = np.squeeze(states)
    next_states = np.squeeze(next_states)

    targets = rewards + gamma * (np.amax(model.predict_on_batch(next_states), axis=1)) * (1 - dones)
    targets_full = model.predict_on_batch(states)

    ind = np.array([i for i in range(batch_size)])

    targets_full[[ind], [actions]] = targets

    model.fit(states, targets_full, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def train_dqn(episode):
    loss = []
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 500
        for i in range(max_steps):

            action = act(state)
            env.render()

            next_state, reward, done, _ = env.step(action)
            score += reward

            next_state = np.reshape(next_state, (1, 8))
            remember(state, action, reward, next_state, done)
            state = next_state
            replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss

if __name__ == '__main__':
    print(env.observation_space)
    print(env.action_space)
    episodes = 400
    loss = train_dqn(episodes)
    plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
    plt.show()