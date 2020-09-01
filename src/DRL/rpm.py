# from collections import deque
import numpy as np
import random
import torch
import pickle as pickle

class rpm(object):
    # replay memory
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size, device, only_state=False):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)            
            return res.to(device)
        else:
            item_count = 5
            res = []
            for i in range(5):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                res.append(k.to(device))
            return res[0], res[1], res[2], res[3], res[4]
