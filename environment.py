import numpy as np
import matplotlib.pyplot as plt
import torch

from tiered_algorithm import state_to_actions
from variables import Actions


device = torch.device("cpu")

def set_device(_device):
    global device
    device = _device

class maze_game:
    def __init__(self, maze_list, max_steps=100, max_rounds=100, verbose=False):
        self.maze_list = maze_list
        self.verbose = verbose
        self.max_steps = max_steps
        self.state = {}
    
    @staticmethod
    def state_to_channels(state):
        c1 = torch.from_numpy(state['grid']).to(device)
        c2 = torch.zeros(state['grid'].shape).to(device)
        c3 = torch.zeros(state['grid'].shape).to(device)
        
        c2[tuple(state['agent'])] = 1.0
        c3[tuple(state['target'])] = 1.0

        envstate = torch.stack([c1, c2, c3]).float()
        
        return envstate
    
    def get_state(self):
        return self.state_to_channels(self.state)
    
    def reset(self):
        self.map_number = np.random.randint(0,len(self.maze_list))
        self.state['grid'] = self.maze_list[self.map_number]
        
        self.state['target'] = np.random.randint(0, self.state['grid'].shape[-1], 2)
        while self.state['grid'][tuple(self.state['target'])]:
            self.state['target'] = np.random.randint(0, self.state['grid'].shape[-1], 2)
        
        self.state['agent'] = np.random.randint(0, self.state['grid'].shape[-1], 2)
        while self.state['grid'][tuple(self.state['agent'])] or np.all(self.state['target']==self.state['agent']):
            self.state['agent'] = np.random.randint(0, self.state['grid'].shape[-1], 2)
        
        self.hist = [self.state['agent'].tolist()]
        self.steps = 0
        self.rounds = 0
        pass
    
    def set_position(self, pos):
        self.state['agent'] = pos
        
    def set_target(self, pos):
        self.state['target'] = pos
    
    def step(self, actions):
        reward_total = 0
        self.rounds += 1
        
        reward = -0.04
        
        for action in actions:
            self.steps += 1
            if action.value == 4:
                reward = -0.04
                reward_total += reward
                continue
                
            dx = np.sin(action.value * np.pi/2).astype(int)
            dy = -np.cos(action.value * np.pi/2).astype(int)

            invalid = False
            if self.state['agent'][0] + dy < 0 or self.state['agent'][0] + dy >= self.state['grid'].shape[0]:
                invalid = True
            elif self.state['agent'][1] + dx < 0 or self.state['agent'][1] + dx >= self.state['grid'].shape[-1]:
                invalid = True
            elif self.state['grid'][self.state['agent'][0]+dy, self.state['agent'][1]+dx]:
                invalid = True

            if not invalid:
                self.state['agent'][1] += dx
                self.state['agent'][0] += dy

            if self.state['agent'].tolist() in self.hist:
                reward = -0.25
            if invalid:
                reward = -0.75
            elif np.all(self.state['target']==self.state['agent']):
                reward = 1.0

            self.hist.append(self.state['agent'].tolist())

        
            reward_total += reward
            if self.steps > self.max_steps or np.all(self.state['target']==self.state['agent']): break
                
        status = 'not_over'
        if self.steps > self.max_steps:
            status = 'lose'
        elif np.all(self.state['target']==self.state['agent']):
            status = 'win'
        
        if reward != 1:
            reward_total -= 0.2
        
        # state reward done win
        return self.state_to_channels(self.state), reward_total, status!='not_over', status=='win'

    def observe(self):
        return self.state
    
    def valid_actions(self):
        valid = []
        if not self.state['agent'][0] + 1 >= self.state['grid'].shape[0] and self.state['grid'][self.state['agent'][0]+1, self.state['agent'][1]+0] == 0:
            valid.append(Actions.SOUTH)
        if not self.state['agent'][0] - 1 < 0 and self.state['grid'][self.state['agent'][0]-1, self.state['agent'][1]+0] == 0:
            valid.append(Actions.NORTH)
        if not self.state['agent'][1] + 1 >= self.state['grid'].shape[1] and self.state['grid'][self.state['agent'][0]+0, self.state['agent'][1]+1] == 0:
            valid.append(Actions.EAST)
        if not self.state['agent'][1] - 1 < 0 and self.state['grid'][self.state['agent'][0]-0, self.state['agent'][1]-1] == 0:
            valid.append(Actions.WEST)
            
        return valid
    
    def best_seq(self, n_actions=4):
        actions = state_to_actions(self.state, limit_n=n_actions)
        
        actions += [Actions.NONE] * (n_actions-len(actions))
        
        return torch.tensor([i.value for i in actions]).view(n_actions, 1, 1).to(device)
    
    def is_complete(self):
        return np.all(self.state['target']==self.state['agent']) or self.steps > self.max_steps
    
    def show(self):
        plt.grid('on')
        nrows, ncols = self.state['grid'].shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = 1-self.state['grid'].copy()
        for row,col in self.hist:
            canvas[row,col] = 0.6

        if self.state['agent'] is not None:
            canvas[tuple(self.state['agent'])] = 0.3
        if self.state['target'] is not None:
            canvas[tuple(self.state['target'])] = 0.8
        img = plt.imshow(canvas, interpolation='none', cmap='gray')
        return img