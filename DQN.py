import time

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from collections import deque
import random
import datetime
import os
from atari_wrappers import wrap_dqn

class DQN(nn.Module):
    '''
    Deep Q-Network
    '''
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, inputs):
        '''
        Forward propogation
        
        inputs: images. expected sshape is (batch_size, frames, width, height)
        '''
        out = F.relu(self.conv1(inputs))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out
    
class ReplayMemory():
    '''
    Replay memory to store states, actions, rewards, dones for batch sampling
    '''
    def __init__(self, capacity):
        '''
        capacity: replay memory capacity
        '''
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, done, next_state):
        '''
        state: current state, atari_wrappers.LazyFrames object
        action: action
        reward: reward for the action
        done: "done" flag is True when the episode finished
        next_state: next state, atari_wrappers.LazyFrames object
        '''
        experience = (state, action, reward, done, next_state)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        '''
        Samples the data from the buffer of a desired size
        
        batch_size: sample batch size
        return: batch of (states, actions, rewards, dones, next states).
                 all are numpy arrays. states and next states have shape of 
                 (batch_size, frames, width, height), where frames = 4.
                 actions, rewards and dones have shape of (batch_size,)
        '''
        if self.count() < batch_size:
            batch = random.sample(self.buffer, self.count())
        else:
            batch = random.sample(self.buffer, batch_size)
            
        state_batch = np.array([np.array(experience[0]) for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        done_batch = np.array([experience[3] for experience in batch])
        next_state_batch = np.array([np.array(experience[4]) for experience in batch])
        
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch
    
    def count(self):
        return len(self.buffer)