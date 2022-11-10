import ray
from ray import tune
from ray.rllib import agents
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = tune.run(
    'DQN', 
    stop={
        'timesteps_total': 50000
    },
    config={
    "env": 'LunarLander-v2',
    "num_workers": 3,
    "gamma" : tune.grid_search([0.999, 0.8]),
    "lr": tune.grid_search([1e-2, 1e-3, 1e-4]),
    }
)

sns.set()

ray.init()
config = {'gamma': 0.999,
          'lr': 0.0001,
          "n_step": 1000,
          'num_workers': 3,
          'monitor': True}
trainer2 = agents.dqn.DQNTrainer(env='LunarLander-v2', config=config)
results2 = trainer2.train()