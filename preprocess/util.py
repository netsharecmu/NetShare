import pickle, ipaddress, copy, time, more_itertools, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from field import *
from output import *
from tqdm import tqdm


# IP: 32 bits, port: 16 bits
def Decimal2Binary(decimal, num_bits):
    binary = bin(decimal)[2:].zfill(num_bits)

    binary = [int(b) for b in binary]

    bits = []

    for b in binary:
        if b == 0:
            bits += [1.0, 0.0]

        elif b == 1:
            bits += [0.0, 1.0]

        else:
            print("Binary number is zero or one!")

    return bits



# norm_option
# 0: ZERO_ONE
# 1: MINUSONE_ONE
def col_norm(col, norm_option):
    if norm_option == Normalization.ZERO_ONE:
        return (col - col.min()) / float(col.max() - col.min())

    elif norm_option == Normalization.MINUSONE_ONE:
        return 2 * (col - col.min()) / float(col.max() - col.min()) - 1

    else:
        print("Unknown normalization!")



def plot_cdf(count_list, xlabel, ylabel, title, base_dir):
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

    plt.plot(x, cdf)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(os.path.join(base_dir, title+".png"), dpi=300)

    # print("{}\t{}".format(xlabel, "CDF"))
    # for index, key in enumerate(list(dist_dict.keys())[:100]):
    #     print(key, cdf[index])

    # plt.show()


# Split list *a* into *n* chunks evenly
def chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# Yield successive n-sized chunks from l. 
def divide_chunks(l, n):
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


