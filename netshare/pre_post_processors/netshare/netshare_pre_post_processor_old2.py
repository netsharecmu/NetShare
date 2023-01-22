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

EPS = 1e-8


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
                if "word2vec" in getattr(m, 'encoding', '')] + \
            [t for t in self._config["timeseries"]
                if "word2vec" in getattr(t, 'encoding', '')]
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
                    annoy_n_trees=self._config["word2vec"]["annoy_n_trees"]
                )
                word2vec_model = Word2Vec.load(word2vec_model_path)

        # Create field instances
        fields = {}  # Python 3.6+ supports order-preserving dictionary
        for field in self._config["metadata"] + self._config["timeseries"]:
            if not isinstance(field.column, str):
                raise ValueError('"column" should be a string')
            field_name = getattr(field, 'name', field.column)

            # BitField
            if 'bit' in getattr(field, 'encoding', ''):
                if not getattr(field, 'n_bits', False):
                    raise ValueError(
                        "`n_bits` needs to be specified for bit fields")
                field_instance = BitField(
                    name=field_name,
                    num_bits=field.n_bits
                )

            # word2vec field
            if 'word2vec' in getattr(field, 'encoding', ''):
                field_instance = ContinuousField(
                    name=field_name,
                    norm_option=Normalization.MINUSONE_ONE,  # l2-norm
                    dim_x=self._config["word2vec"]["vec_size"]
                )

            # float - Continuous Field
            if field.type == "float":
                field_instance = ContinuousField(
                    name=field_name,
                    norm_option=getattr(Normalization, field.normalization),
                    min_x=min(df[field.column])-EPS,
                    max_x=max(df[field.column])+EPS,
                    dim_x=1
                )
                if getattr(field, 'log1p_norm', False):
                    df[field.column] = np.log1p(df[field.column])

            # string - Categorical Field
            if field.type == "string" and 'encoding' not in field:
                field_instance = DiscreteField(
                    choices=list(set(df[field.column])),
                    name=getattr(field, 'name', field.column))

            fields[field_name] = field_instance

        # Multi-chunk related field instances
        # n_chunk=1 reduces to plain DoppelGANger
        if self._config["n_chunks"] > 1:
            fields["startFromThisChunk"] = DiscreteField(
                name="startFromThisChunk",
                choices=[0.0, 1.0]
            )

            for chunk_id in range(self._config["n_chunks"]):
                fields["chunk_{}".format(chunk_id)] = DiscreteField(
                    name="chunk_{}".format(chunk_id),
                    choices=[0.0, 1.0]
                )

        # Timestamp
        if self._config["timestamp"]["generation"] and \
                self._config["timestamp"]["column"]:
            if self._config["timestamp"]["encoding"] == "interarrival":
                fields["flow_start"] = ContinuousField(
                    name="flow_start",
                    norm_option=getattr(
                        Normalization, self._config["timestamp"].normalization)
                )
                fields["interarrival_within_flow"] = ContinuousField(
                    name="interarrival_within_flow",
                    norm_option=getattr(
                        Normalization, self._config["timestamp"].normalization)
                )
            elif self._config["timestamp"]["encoding"] == "raw":
                field_name = getattr(
                    self._config["timestamp"], "name", self._config["timestamp"]["column"])
                fields[field_name] = ContinuousField(
                    name=field_name,
                    norm_option=getattr(
                        Normalization, self._config["timestamp"].normalization)
                )
            else:
                raise ValueError("Timestamp encoding can be only \
                `interarrival` or 'raw")

        print("Fields:", fields)

        # Group data by metadata and measurement
        gk = df.groupby(by=[m.column for m in self._config["metadata"]])

        # flow size distribution
        plot_cdf(
            count_list=sorted(gk.size().values, reverse=True),
            xlabel="# of packets per flow",
            ylabel="CDF",
            title="",
            filename="raw_pkts_per_flow.png",
            base_dir=output_folder
        )

        '''generating cross-chunk flow stats'''
        # split big df to chunks
        print("Using {}".format(self._config["df2chunks"]))
        df_chunks, _tmp_sizeortime = df2chunks(
            big_raw_df=df,
            config_timestamp=self._config["timestamp"],
            split_type=self._config["df2chunks"],
            n_chunks=self._config["n_chunks"])

        df_chunk_cnt_validation = 0
        for chunk_id, df_chunk in enumerate(df_chunks):
            print("Chunk_id: {}, # of pkts/records: {}".format(
                chunk_id, len(df_chunk)))
            df_chunk_cnt_validation += len(df_chunk)
        print("df_chunk_cnt_validation:", df_chunk_cnt_validation)

        if self._config["df2chunks"] == "fixed_size":
            chunk_size = _tmp_sizeortime
            print("Chunk size: {} records \
                (pkts for PCAP, # of records for NetFlow)".format(
                chunk_size))
        elif self._config["df2chunks"] == "fixed_time":
            chunk_time = _tmp_sizeortime
            print("Chunk time: {} seconds".format(chunk_time / (10**6)))
        else:
            raise ValueError("Unknown df2chunks type!")

        flowkeys_chunkidx_file = os.path.join(
            output_folder, "flowkeys_idxlist.json")
        print("compute flowkey-chunk list from scratch...")
        flow_chunkid_keys = {}
        for chunk_id, df_chunk in enumerate(df_chunks):
            gk = df_chunk.groupby([m.column for m in self._config["metadata"]])
            flow_keys = list(gk.groups.keys())
            flow_keys = list(map(str, flow_keys))
            flow_chunkid_keys[chunk_id] = flow_keys

        # key: flow key
        # value: [a list of appeared chunk idx]
        flowkeys_chunkidx = {}
        for chunk_id, flowkeys in flow_chunkid_keys.items():
            print("processing chunk {}/{}, # of flows: {}".format(
                chunk_id + 1, len(df_chunks), len(flowkeys)))
            for k in flowkeys:
                if k not in flowkeys_chunkidx:
                    flowkeys_chunkidx[k] = []
                flowkeys_chunkidx[k].append(chunk_id)

        with open(flowkeys_chunkidx_file, 'w') as f:
            json.dump(flowkeys_chunkidx, f)

        num_non_continuous_flows = 0
        num_flows_cross_chunk = 0
        flowkeys_chunklen_list = []
        for flowkey, chunkidx in flowkeys_chunkidx.items():
            flowkeys_chunklen_list.append(len(chunkidx))
            if not continuous_list_flag(chunkidx):
                num_non_continuous_flows += 1
            if len(chunkidx) > 1:
                num_flows_cross_chunk += 1

        print("# of total flows:", len(flowkeys_chunklen_list))
        print("# of total flows (sanity check):", len(
            df.groupby([m.column for m in self._config["metadata"]])))
        print("# of flows cross chunk (of total flows): {} ({}%)".format(
            num_flows_cross_chunk,
            float(num_flows_cross_chunk) / len(flowkeys_chunklen_list) * 100))
        print("# of non-continuous flows:", num_non_continuous_flows)
        plot_cdf(
            count_list=flowkeys_chunklen_list,
            xlabel="# of chunks per flow",
            ylabel="CDF",
            title="",
            filename="raw_flow_numchunks.png",
            base_dir=output_folder
        )

        # global_max_flow_len for consistency between chunks
        per_chunk_flow_len_agg = []
        max_flow_lens = []
        for chunk_id, df_chunk in enumerate(df_chunks):
            # corner case: skip for empty df_chunk
            if len(df_chunk) == 0:
                continue
            gk_chunk = df_chunk.groupby(
                by=[m.column for m in self._config["metadata"]])
            max_flow_lens.append(max(gk_chunk.size().values))
            per_chunk_flow_len_agg += list(gk_chunk.size().values)
            print("chunk_id: {}, max_flow_len: {}".format(
                chunk_id, max(gk_chunk.size().values)))
        if not self._config["max_flow_len"]:
            global_max_flow_len = max(max_flow_lens)
        else:
            global_max_flow_len = self._config["max_flow_len"]
        print("global max flow len:", global_max_flow_len)
        print("Top 10 per-chunk flow length:",
              sorted(per_chunk_flow_len_agg)[-10:])

        return True

    def _post_process(self, input_folder, output_folder,
                      pre_processed_data_folder, log_folder):
        print(f"{self.__class__.__name__}.{inspect.stack()[0][3]}")

        return True
