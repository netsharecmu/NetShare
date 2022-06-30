'''Created on Dec 13, 2021. Helper functions for preprocess'''

import pickle, ipaddress, copy, time, more_itertools, os, math, json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tqdm import tqdm
from gensim.models import Word2Vec, word2vec
from sklearn import preprocessing

from util import *
from field import *
from output import *
from word2vec_embedding import get_vector

def countList2cdf(count_list):
    # dist_dict: {key : count}
    dist_dict = {}
    for x in count_list:
        if x not in dist_dict:
            dist_dict[x] = 0
        dist_dict[x] += 1

    dist_dict = {k: v for k, v in sorted(dist_dict.items(), key = lambda x: x[0])}
    x = dist_dict.keys()
    # print(sum(dist_dict.values()))
    pdf = np.asarray(list(dist_dict.values()), dtype=float) / float(sum(dist_dict.values()))
    cdf = np.cumsum(pdf)

    return x, cdf

# l: [1, 2, 3, 4]: True
# [1, 3, 5, 7]: False
def continuous_list_flag(l):
    first_order_diff = np.diff(l)
    return len(set(first_order_diff)) <= 1

# flg_str: .AP...
def netflow_flag_str2bits(flg_str):
    res = []
    for idx, s in enumerate(flg_str):
        if s == ".":
            res.append(0)
        else:
            res.append(1)
    
    return res

def netflow_flag_str2int(flg_str):
    res = 0
    for idx, s in enumerate(flg_str):
        if s != ".":
            res += 2**(5-idx)
    
    return res

# print(netflow_flag_str2bits(".AP..."))
# print(netflow_flag_str2int(".AP..."))