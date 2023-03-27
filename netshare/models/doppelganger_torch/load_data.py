import os
import math
import numpy as np
import pickle


def load_data(path, sample_len, flag="train"):

    data_npz = np.load(os.path.join(path, "data_{}.npz".format(flag)))
    with open(os.path.join(path, "data_feature_output.pkl"), "rb") as f:
        data_feature_outputs = pickle.load(f)
    with open(os.path.join(path, "data_attribute_output.pkl"), "rb") as f:
        data_attribute_outputs = pickle.load(f)

    data_feature = data_npz["data_feature"]
    data_attribute = data_npz["data_attribute"]
    data_gen_flag = data_npz["data_gen_flag"]

    # Append data_feature and data_gen_flag to multiple of sample_len
    timeseries_len = data_feature.shape[1]
    ceil_timeseries_len = math.ceil(timeseries_len / sample_len) * sample_len
    data_feature = np.pad(
        data_feature,
        pad_width=((0, 0),
                   (0, ceil_timeseries_len - timeseries_len),
                   (0, 0)),
        mode='constant', constant_values=0)
    data_gen_flag = np.pad(
        data_gen_flag,
        pad_width=((0, 0),
                   (0, ceil_timeseries_len - timeseries_len)),
        mode='constant', constant_values=0)

    return (
        data_feature,
        data_attribute,
        data_gen_flag,
        data_feature_outputs,
        data_attribute_outputs,
    )
