from classes_and_functions.dqn_agent import DQNAgent
from classes_and_functions.replay_buffer import ReplayBuffer
import gymnasium as gym
import random
import numpy as np
import torch
import sys


def initialize_agent_and_replay_buffer(config=None):
    
    if config is None:
        sys.exit("Error: No configuration provided for initializing agent and replay buffer")

    env = gym.make(config["ENVIRONMENT"])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    config["STATE_SIZE"] = state_size
    config["ACTION_SIZE"] = action_size

    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    
    dqn_agent = DQNAgent(config=config)
    replay_buffer = ReplayBuffer(config["BUFFER_SIZE"])
    return dqn_agent,replay_buffer