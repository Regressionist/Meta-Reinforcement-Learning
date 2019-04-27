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
from DQN import DQN, ReplayMemory



class Agent(object):
    '''
    Implements training and testing methods
    '''
    def __init__(self):
        self.env=wrap_dqn(gym.make('BreakoutDeterministic-v4'))
        self.num_actions=self.env.action_space.n
        self.dqn=DQN(self.num_actions).cuda()
        self.target_dqn=DQN(self.num_actions).cuda()
        
        self.buffer=ReplayMemory(150000)
        self.gamma=0.99
        
        self.optimizer=optim.Adam(self.dqn.parameters())
        self.out_dir='/scratch/ab8084/atari/saved/'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        
        self.reward_episodes=[]
        self.lengths_episodes=[]
        self.benchmark=0
        
    def to_var(self,x):
        '''
        Converts torch tensor x to torch variable
        '''
        return Variable(x).cuda()
    
    def predict_q_values(self,states):
        '''
        Computes q values of states by passing them through the behavior network
        states: numpy array, shape is (batch_size,frames,width,height)
        returns actions: shape is (batch_size, num_actions)
        '''
        states=self.to_var(torch.from_numpy(states).float())
        actions=self.dqn(states)
        return actions
        
    def predict_q_target_values(self,states):
        '''
        Computes q values of next states by passing them through the target network
        states: numpy array, shape is (batch_size,frames,width,height)
        returns actions: shape is (batch_size, num_actions)
        '''
        states=self.to_var(torch.from_numpy(states).float())
        actions=self.target_dqn(states)
        return actions
    
    def select_action(self,state,epsilon):
        choice = np.random.choice([0, 1], p=(epsilon, (1 - epsilon)))
        
        if choice==0:
            return np.random.choice(range(self.num_actions))
        else:
            state=np.expand_dims(state,0)
            actions=self.predict_q_values(state)
            return np.argmax(actions.data.cpu().numpy())
        
    def update(self,states,targets,actions):
        '''
        Calculates loss and updates the weights of the behavior network using backprop
        states: numpy array, shape is (batch_size,frames,width,height)
        actions: numpy array, shape is(batch_size,num_actions)
        targets: numpy array, shape is (batch_size)
        '''
        targets = self.to_var(torch.unsqueeze(torch.from_numpy(targets).float(), -1))
        actions = self.to_var(torch.unsqueeze(torch.from_numpy(actions).long(), -1))
        predicted_values = self.predict_q_values(states)
        affected_values = torch.gather(predicted_values, 1, actions)
        loss = F.smooth_l1_loss(affected_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def calculate_q_targets(self,next_states,rewards,dones):
        '''
        Calculate targets from target network
        next_states: numpy array, shape is (batch_size, frames, width, height)
        rewards: numpy array, shape is (batch_size,)
        dones: numpy array, shape is (batch_size,)
        '''
        dones_mask = (dones == 1)
        predicted_q_target_values = self.predict_q_target_values(next_states)
        next_max_q_values = np.max(predicted_q_target_values.data.cpu().numpy(), axis=1)
        next_max_q_values[dones_mask] = 0 
        q_targets = rewards + self.gamma * next_max_q_values
        return q_targets
    
    def sync_target_network(self):
        '''
        Copies weights from estimation to target network
        '''
        primary_params = list(self.dqn.parameters())
        target_params = list(self.target_dqn.parameters())
        for i in range(0, len(primary_params)):
            target_params[i].data[:] = primary_params[i].data[:]
            
    def play(self,episodes):
        '''
        plays for epsiodes number of episodes
        '''
        for i in range(1,episodes+1):
            done=False
            state=self.env.reset()
            while not done:
                action = self.select_action(state, 0)
                state, reward, done, _ = self.env.step(action)
                self.env.render()
                
    def close_env(self):
        '''
        Clean up
        '''
        self.env.close()
        
    def get_epsilon(self,total_steps,max_epsilon_steps, epsilon_start, epsilon_final):
        return max(epsilon_final, epsilon_start - total_steps / max_epsilon_steps)
    
    def save_final_model(self):
        '''
        Saves final model to the disk
        '''
        filename = '{}/final_model.pth'.format(self.out_dir)
        torch.save({
                'model_state_dict': self.dqn.state_dict(),
                'reward':self.reward_episodes,
                'benchmark':self.benchmark,
                'lengths':self.lengths_episodes
                }, filename)
        

    def save_model_during_training(self, episode):
        '''
        Saves temporary models to the disk during training
        
        :param episode: episode number
        '''
        filename = '{}/current_model_{}.pth'.format(self.out_dir, episode)
        torch.save(self.dqn.state_dict(), filename)
        
        
    def load_model(self, filename):
        '''
        Loads model from the disk
        
        :param filename: model filename
        '''
        try:
            checkpoint = torch.load('/scratch/ab8084/atari/saved/final_model.pth')
            self.dqn.load_state_dict(checkpoint['model_state_dict'])
            self.reward_episodes=checkpoint['reward']
            self.lengths_episodes=checkpoint['lengths']
            self.benchmark=checkpoint['benchmark']
        except:   
            self.dqn.load_state_dict(torch.load(filename))
                                        
        self.sync_target_network()
    
    def train(self, replay_buffer_fill_len, batch_size, episodes, stop_reward,max_epsilon_steps, epsilon_start, epsilon_final, sync_target_net_freq):
        '''
        replay_buffer_fill_len: how many elements should replay buffer contain before training starts
        batch_size: batch size
        episodes: how many episodes (max. value) to iterate
        stop_reward: running reward value to be reached. upon reaching that value the training is stoped
        max_epsilon_steps: maximum number of epsilon steps
        epsilon_start: start epsilon value
        epsilon_final: final epsilon value, effectively a limit
        sync_target_net_freq: how often to sync estimation and target networks
        '''
        
        start_time=time.time()
        print('Start training at: '+ time.asctime(time.localtime(start_time)))
        
        total_steps = 0
        running_episode_reward = 0
        
        
        print('Populating Replay Buffer')
        print('\n')
        
        state=self.env.reset()
        for i in range(replay_buffer_fill_len):
            done=False
            action=self.select_action(state,0.5)
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add(state, action, reward, done, next_state)
            state = next_state
            if done:
                state=self.env.reset()
                
        print('Replay Buffer populated with {} transitions, starting training...'.format(self.buffer.count()))
        print('\n')
        
        for i in range(1,episodes+1):
            done=False
            state=self.env.reset()
              
            episode_reward=0
            episode_length=0
              
            while not done:
                if (total_steps % sync_target_net_freq) == 0:
                    print('synchronizing target network...')
                    #print('\n')
                    self.sync_target_network()
                    
                epsilon = self.get_epsilon(total_steps, max_epsilon_steps, epsilon_start, epsilon_final)
                action = self.select_action(state, epsilon)
              
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.add(state, action, reward, done, next_state)
                s_batch, a_batch, r_batch, d_batch, next_s_batch = self.buffer.sample(batch_size)
                q_targets = self.calculate_q_targets(next_s_batch, r_batch, d_batch)
              
                self.update(s_batch, q_targets, a_batch)
              
                state = next_state
                total_steps += 1
                episode_length += 1
                episode_reward += np.sign(reward)
                
            self.reward_episodes.append(episode_reward)
            self.lengths_episodes.append(episode_length)
              
            running_episode_reward = running_episode_reward * 0.9 + 0.1 * episode_reward
            if (i % 50) == 0 or (running_episode_reward > stop_reward):
                print('global step: {}'.format(total_steps) , ' | episode: {}'.format(i) , ' | mean episode_length: {}'.format(np.mean(self.lengths_episodes[-50:])) , ' | mean episode reward: {}'.format(np.mean(self.reward_episodes[-50:])))
                #print('episode: {}'.format(i))
                #print('current epsilon: {}'.format(round(epsilon, 2)))
                #print('mean episode_length: {}'.format(np.mean(lengths_episodes[-50:])))
                #print('mean episode reward: {}'.format(np.mean(reward_episodes[-50:])))
                #print('\n')
                if np.mean(self.reward_episodes[-50:])>self.benchmark:
                    self.save_final_model()
                    self.benchmark=np.mean(self.reward_episodes[-50:])
            if running_episode_reward > stop_reward:
                print('stop reward reached!')
                print('saving final model...')
                print('\n')
                #self.save_final_model()
                break
              
        print('Finish training at: '+ time.asctime(time.localtime(start_time)))
              
        