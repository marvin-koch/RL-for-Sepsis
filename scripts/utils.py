"""
Collection of utilities used for model learning and evaluation.

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================

Forked from https://github.com/acneyouth1996/RL-for-sepsis-continuous/blob/yong_v0/

Modified such that buffer can additonnaly save SOFA subscores
"""

import logging
import numpy as np
from hashlib import sha1
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.common import mask_from_lengths
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)

class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

def one_hot(x, num_x, data_type='numpy', device=None):
    if data_type == 'numpy':
        res = np.zeros(num_x)
    elif data_type == 'torch':
        res = torch.zeros(num_x).to(device)
    res[x] = 1.0
    return res

                 

"""
Generic replay buffer for standard gym tasks (adapted from https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/utils.py)
Modified such that me sample consecutive state,action,reward pairs (RNNs are time senstive)
"""
class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device, encoded_state=False):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.encoded_state = encoded_state

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action_c = np.zeros((self.max_size, 2))
        self.next_action_c = np.zeros((self.max_size, 2))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.score = np.zeros((self.max_size, 3))
        self.sofa1 =  np.zeros((self.max_size, 1))
        self.sofa2 =  np.zeros((self.max_size, 1))
        self.sofa3 =  np.zeros((self.max_size, 1))
        self.sofa4 =  np.zeros((self.max_size, 1))
        self.sofa5 =  np.zeros((self.max_size, 1))
        self.sofa6 =  np.zeros((self.max_size, 1))

        self.sofa1_next =  np.zeros((self.max_size, 1))
        self.sofa2_next =  np.zeros((self.max_size, 1))
        self.sofa3_next =  np.zeros((self.max_size, 1))
        self.sofa4_next =  np.zeros((self.max_size, 1))
        self.sofa5_next =  np.zeros((self.max_size, 1))
        self.sofa6_next =  np.zeros((self.max_size, 1))
   
   
        self.next_score = np.zeros((self.max_size, 3))
        self.outcome = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))



    def add(self, state,  action_c, next_action_c, next_state, reward, score, next_score, outcome, done):
        """
        Add episode
        """
        self.state[self.ptr] = state
        self.action_c[self.ptr] = action_c
        self.next_action_c[self.ptr] = next_action_c
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.score[self.ptr] = score[:3]
        self.sofa1[self.ptr] = (score[3])
        self.sofa2[self.ptr] = (score[4])
        self.sofa3[self.ptr] = (score[5])
        self.sofa4[self.ptr] = (score[6])
        self.sofa5[self.ptr] = (score[7])
        self.sofa6[self.ptr] = (score[8])

        self.next_score[self.ptr] = next_score[:3]
        self.sofa1_next[self.ptr] = (next_score[3])
        self.sofa2_next[self.ptr] = (next_score[4])
        self.sofa3_next[self.ptr] = (next_score[5])
        self.sofa4_next[self.ptr] = (next_score[6])
        self.sofa5_next[self.ptr] = (next_score[7])
        self.sofa6_next[self.ptr] = (next_score[8])


        self.outcome[self.ptr] = outcome
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self, extra=False):
        """
        Sample episode
        """
        valid = False
        while not valid:
            ind = np.random.randint(0, self.crt_size)
            e = ind + self.batch_size
            if e < self.crt_size:
                valid = True
            ind = [i for i in range(ind, e)]

        
        if extra:
            return(
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action_c[ind]).to(self.device),
                torch.FloatTensor(self.next_action_c[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.score[ind]).to(self.device),
                torch.FloatTensor(self.sofa1[ind]).to(self.device),
                torch.FloatTensor(self.sofa2[ind]).to(self.device),
                torch.FloatTensor(self.sofa3[ind]).to(self.device),
                torch.FloatTensor(self.sofa4[ind]).to(self.device),
                torch.FloatTensor(self.sofa5[ind]).to(self.device),
                torch.FloatTensor(self.sofa6[ind]).to(self.device),
                torch.FloatTensor(self.next_score[ind]).to(self.device),
                torch.FloatTensor(self.sofa1_next[ind]).to(self.device),
                torch.FloatTensor(self.sofa2_next[ind]).to(self.device),
                torch.FloatTensor(self.sofa3_next[ind]).to(self.device),
                torch.FloatTensor(self.sofa4_next[ind]).to(self.device),
                torch.FloatTensor(self.sofa5_next[ind]).to(self.device),
                torch.FloatTensor(self.sofa6_next[ind]).to(self.device),
                torch.FloatTensor(self.outcome[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )
        else:
            return(
                    torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.FloatTensor(self.action_c[ind]).to(self.device),
                    torch.FloatTensor(self.next_action_c[ind]).to(self.device),
                    torch.FloatTensor(self.next_state[ind]).to(self.device),
                    torch.FloatTensor(self.reward[ind]).to(self.device),
                    torch.FloatTensor(self.score[ind]).to(self.device),
                    torch.FloatTensor(self.next_score[ind]).to(self.device),
                    torch.FloatTensor(self.outcome[ind]).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device)
                )
        

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action_c.npy", self.action_c[:self.crt_size])
        np.save(f"{save_folder}_next_action_c.npy", self.next_action_c[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_score.npy", self.score[:self.crt_size])
        np.save(f"{save_folder}_sofa1.npy", self.sofa1[:self.crt_size])
        np.save(f"{save_folder}_sofa2.npy", self.sofa2[:self.crt_size])
        np.save(f"{save_folder}_sofa3.npy", self.sofa3[:self.crt_size])
        np.save(f"{save_folder}_sofa4.npy", self.sofa4[:self.crt_size])
        np.save(f"{save_folder}_sofa5.npy", self.sofa5[:self.crt_size])
        np.save(f"{save_folder}_sofa6.npy", self.sofa6[:self.crt_size])
        np.save(f"{save_folder}_next_score.npy", self.next_score[:self.crt_size])
        np.save(f"{save_folder}_sofa1n.npy", self.sofa1_next[:self.crt_size])
        np.save(f"{save_folder}_sofa2n.npy", self.sofa2_next[:self.crt_size])
        np.save(f"{save_folder}_sofa3n.npy", self.sofa3_next[:self.crt_size])
        np.save(f"{save_folder}_sofa4n.npy", self.sofa4_next[:self.crt_size])
        np.save(f"{save_folder}_sofa5n.npy", self.sofa5_next[:self.crt_size])
        np.save(f"{save_folder}_sofa6n.npy", self.sofa6_next[:self.crt_size])
        np.save(f"{save_folder}_outcome.npy", self.outcome[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)
        
        
    def load(self, save_folder, size=-1, bootstrap=False):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action_c[:self.crt_size] = np.load(f"{save_folder}_action_c.npy")[:self.crt_size]
        self.next_action_c[:self.crt_size] = np.load(f"{save_folder}_next_action_c.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.score[:self.crt_size] = np.load(f"{save_folder}_score.npy")[:self.crt_size]
        self.sofa1[:self.crt_size] = np.load(f"{save_folder}_sofa1.npy")[:self.crt_size]
        self.sofa2[:self.crt_size] = np.load(f"{save_folder}_sofa2.npy")[:self.crt_size]
        self.sofa3[:self.crt_size] = np.load(f"{save_folder}_sofa3.npy")[:self.crt_size]
        self.sofa4[:self.crt_size] = np.load(f"{save_folder}_sofa4.npy")[:self.crt_size]
        self.sofa5[:self.crt_size] = np.load(f"{save_folder}_sofa5.npy")[:self.crt_size]
        self.sofa6[:self.crt_size] = np.load(f"{save_folder}_sofa6.npy")[:self.crt_size]
 
        self.next_score[:self.crt_size] = np.load(f"{save_folder}_next_score.npy")[:self.crt_size]
        self.sofa1_next[:self.crt_size] = np.load(f"{save_folder}_sofa1n.npy")[:self.crt_size]
        self.sofa2_next[:self.crt_size] = np.load(f"{save_folder}_sofa2n.npy")[:self.crt_size]
        self.sofa3_next[:self.crt_size] = np.load(f"{save_folder}_sofa3n.npy")[:self.crt_size]
        self.sofa4_next[:self.crt_size] = np.load(f"{save_folder}_sofa4n.npy")[:self.crt_size]
        self.sofa5_next[:self.crt_size] = np.load(f"{save_folder}_sofa5n.npy")[:self.crt_size]
        self.sofa6_next[:self.crt_size] = np.load(f"{save_folder}_sofa6n.npy")[:self.crt_size]
        self.outcome[:self.crt_size] = np.load(f"{save_folder}_outcome.npy")[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

        
        
        if bootstrap:
            # Get the indicies of the above arrays that are non-zero
            nonzero_ind = (self.reward !=0)[:,0]
            num_nonzero = sum(nonzero_ind)
            self.state[self.crt_size:(self.crt_size+num_nonzero)] = self.state[nonzero_ind]
            self.action_c[self.crt_size:(self.crt_size+num_nonzero)] = self.action_c[nonzero_ind]
            self.next_action_c[self.crt_size:(self.crt_size+num_nonzero)] = self.next_action_c[nonzero_ind]
            self.next_state[self.crt_size:(self.crt_size+num_nonzero)] = self.next_state[nonzero_ind]
            self.reward[self.crt_size:(self.crt_size+num_nonzero)] = self.reward[nonzero_ind]
            self.score[self.crt_size:(self.crt_size+num_nonzero)] = self.score[nonzero_ind]
            self.sofa1[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa1[nonzero_ind]
            self.sofa2[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa2[nonzero_ind]
            self.sofa3[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa3[nonzero_ind]
            self.sofa4[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa4[nonzero_ind]
            self.sofa5[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa5[nonzero_ind]
            self.sofa6[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa6[nonzero_ind]
            self.sofa1_next[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa1_next[nonzero_ind]
            self.sofa2_next[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa2_next[nonzero_ind]
            self.sofa3_next[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa3_next[nonzero_ind]
            self.sofa4_next[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa4_next[nonzero_ind]
            self.sofa5_next[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa5_next[nonzero_ind]
            self.sofa6_next[self.crt_size:(self.crt_size+num_nonzero)] = self.sofa6_next[nonzero_ind]
            self.next_score[self.crt_size:(self.crt_size+num_nonzero)] = self.next_score[nonzero_ind]
            self.outcome[self.crt_size:(self.crt_size+num_nonzero)] = self.outcome[nonzero_ind]
            self.not_done[self.crt_size:(self.crt_size+num_nonzero)] = self.not_done[nonzero_ind]
            
            
            self.crt_size += num_nonzero

            neg_ind = (self.reward < 0)[:,0]
            num_neg = sum(neg_ind)
            self.state[self.crt_size:(self.crt_size+num_neg)] = self.state[neg_ind]
            self.action_c[self.crt_size:(self.crt_size+num_neg)] = self.action_c[neg_ind]
            self.next_action_c[self.crt_size:(self.crt_size+num_neg)] = self.next_action_c[neg_ind]
            self.next_state[self.crt_size:(self.crt_size+num_neg)] = self.next_state[neg_ind]
            self.reward[self.crt_size:(self.crt_size+num_neg)] = self.reward[neg_ind]
            self.score[self.crt_size:(self.crt_size+num_neg)] = self.score[neg_ind]
            self.sofa1[self.crt_size:(self.crt_size+num_neg)] = self.sofa1[neg_ind]
            self.sofa2[self.crt_size:(self.crt_size+num_neg)] = self.sofa2[neg_ind]
            self.sofa3[self.crt_size:(self.crt_size+num_neg)] = self.sofa3[neg_ind]
            self.sofa4[self.crt_size:(self.crt_size+num_neg)] = self.sofa4[neg_ind]
            self.sofa5[self.crt_size:(self.crt_size+num_neg)] = self.sofa5[neg_ind]
            self.sofa6[self.crt_size:(self.crt_size+num_neg)] = self.sofa6[neg_ind]
            self.sofa1_next[self.crt_size:(self.crt_size+num_neg)] = self.sofa1_next[neg_ind]
            self.sofa2_next[self.crt_size:(self.crt_size+num_neg)] = self.sofa2_next[neg_ind]
            self.sofa3_next[self.crt_size:(self.crt_size+num_neg)] = self.sofa3_next[neg_ind]
            self.sofa4_next[self.crt_size:(self.crt_size+num_neg)] = self.sofa4_next[neg_ind]
            self.sofa5_next[self.crt_size:(self.crt_size+num_neg)] = self.sofa5_next[neg_ind]
            self.sofa6_next[self.crt_size:(self.crt_size+num_neg)] = self.sofa6_next[neg_ind]
            self.next_score[self.crt_size:(self.crt_size+num_neg)] = self.next_score[neg_ind]
            self.outcome[self.crt_size:(self.crt_size+num_neg)] = self.outcome[neg_ind]
            self.not_done[self.crt_size:(self.crt_size+num_neg)] = self.not_done[neg_ind]
            

            self.crt_size += num_neg

            

        print(f"Replay Buffer loaded with {self.crt_size} elements.")


