# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import pickle

from torch.utils.data import Dataset



class TSVDataset(Dataset):
    def __init__(self, path, process_fn, with_heads=True, **kwargs):
        self.process_fn = process_fn
        with open(path, 'r', encoding="GBK") as fin:
            if with_heads:
                self.heads = fin.readline().split('\t')
            else:
                self.heads = None
            self.items = [line.split('\t') for line in fin]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.process_fn(self.items[index])


