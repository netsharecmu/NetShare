import inspect
import pickle
import ipaddress
import copy
import time
import more_itertools
import os
import re
import math
import json
import pickle
import ctypes
import pkgutil
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netshare.ray as ray

from gensim.models import Word2Vec
from tqdm import tqdm

from .util import denormalize, _merge_syn_df, _recalulate_config_ids_in_each_config_group
from .word2vec_embedding import word2vec_train
from .preprocess_helper import countList2cdf, continuous_list_flag, plot_cdf
from .preprocess_helper import df2chunks, split_per_chunk
from ..pre_post_processor import PrePostProcessor
from netshare.utils import Tee, Output, output
from netshare.utils import Normalization
from netshare.utils import ContinuousField, DiscreteField, BitField
from netshare.utils import exec_cmd


class NetsharePrePostProcessor(PrePostProcessor):
    def _pre_process(self, input_folder, output_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        # single file
        if os.path.isfile(input_folder):
            if not input_folder.endswith(".csv"):
                raise ValueError(
                    "Uncompatible file format! "
                    "Please convert your dataset as instructued to supported <csv> formats...")
            print(input_folder)
        # TODO: support multiple files
        else:
            print("Merged file is located at {}")

        if self._config["dataset_type"] == "pcap":
            if input_folder.endswith(".csv"):
                shutil.copyfile(
                    os.path.join(input_folder),
                    os.path.join(output_folder, "raw.csv")
                )
                df = pd.read_csv(input_folder)
            elif input_folder.endswith(".pcap"):
                # compile shared library for converting pcap to csv
                cwd = os.path.dirname(os.path.abspath(__file__))
                cmd = f"cd {cwd} && \
                    cc -fPIC -shared -o pcap2csv.so main.c -lm -lpcap"
                exec_cmd(cmd, wait=True)

                pcap2csv_func = ctypes.CDLL(
                    os.path.join(cwd, "pcap2csv.so")).pcap2csv
                pcap2csv_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                csv_file = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(input_folder))[0] +
                    ".csv")
                pcap2csv_func(
                    input_folder.encode('utf-8'),  # pcap file
                    csv_file.encode('utf-8')  # csv file
                )
                print(f"{input_folder} has been converted to {csv_file}")
                df = pd.read_csv(csv_file)
            else:
                raise ValueError(
                    "PCAP file extension should be `.pcap`(native) or `.csv`(converted)!")
        else:
            if not input_folder.endswith(".csv"):
                raise ValueError(
                    "Non-pcap file, only CSV format is supported!")
            df = pd.read_csv(input_folder)
            df.to_csv(os.path.join(output_folder, "raw.csv"), index=False)

        print("dataset meta type:", self._config["dataset_meta_type"])
        print("dataset type:", self._config["dataset_type"])

        metadata_cols = [m for m in self._config["metadata"]]
        word2vec_cols = \
            [m for m in self._config["metadata"]
                if "word2vec" in m.method] + \
            [t for t in self._config["timeseries"]
                if "word2vec" in t.method]
        print("metadata cols:", [m.column for m in metadata_cols])
        print("word2vec cols:", [w.column for w in word2vec_cols])

        # Word2Vec embedding
        if len(word2vec_cols) == 0:
            print("No word2vec columns... Skipping word2vec embedding...")
        else:
            if self._config["word2vec"]["pretrain_model_path"]:
                print("Loading pretrained `big` Word2Vec model...")
                word2vec_model = Word2Vec.load(
                    self._config["word2vec"]["pretrain_model_path"])
            else:
                word2vec_model_path = word2vec_train(
                    df=df,
                    out_dir=output_folder,
                    model_name=self._config["word2vec"]["model_name"],
                    word2vec_cols=word2vec_cols,
                    word2vec_size=self._config["word2vec"]["vec_size"],
                    annoy_n_trees=self._config["word2vec"]["annoy_n_trees"],
                    force_retrain=True,
                    model_test=True
                )
                word2vec_model = Word2Vec.load(word2vec_model_path)

        return True

    def _post_process(self, input_folder, output_folder,
                      pre_processed_data_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        return True
