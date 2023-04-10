import os
import re
import pickle
import math
import json
import ast
import socket
import struct
import ipaddress
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scapy.all import IP, ICMP, TCP, UDP
from scapy.all import wrpcap
from scipy.stats import rankdata
from .embedding_helper import (
    build_annoy_dictionary_word2vec,
    get_original_obj
)
from .dist_metrics import (
    compute_metrics_netflow_v3,
    compute_metrics_pcap_v3,
    compute_metrics_zeeklog_v3
)
from ...model_managers.netshare_manager.netshare_util import get_configid_from_kv


def convert_sdmetricsConfigQuant_to_fieldValueDict(
    sdmetricsConfigQuant
):
    '''Convert the sdmetricsConfigQuant to fieldValueDict
    Args:
        sdmetricsConfigQuant (dict): returned by create_sdmetrics_config(...,  comparison_type='quantitative')
    Returns:
        fieldValueDict (dict): {field_name: value}
    '''

    fieldValueDict = {}
    for metric_type, metrics in sdmetricsConfigQuant.items():
        for metric_class_name, metric_class in metrics.items():
            # metrics with target (e.g., attr dist similarity)
            if isinstance(metric_class, dict):
                for field_name, field_value in metric_class.items():
                    fieldValueDict[ast.literal_eval(
                        field_name)[0]] = field_value[0][0]
            # metrics without target (e.g., session length)
            elif isinstance(metric_class, list):
                fieldValueDict[metric_class_name] = metric_class[0][0]

    return fieldValueDict


def create_sdmetrics_config(
    config_pre_post_processor,
    comparison_type='both'
):
    # Refer to https://github.com/netsharecmu/SDMetrics_timeseries/blob/master/sdmetrics/reports/timeseries/sunglasses_qr.json to see the format of the config file
    sdmetrics_config = {
        "metadata": {
            "fields": {}
        },
        "config": {
            "metrics": {
                "fidelity": []
            }
        }
    }

    # Enumerate through all the fields in the metadata, timeseries, and timestamp
    for i, field in enumerate(config_pre_post_processor.metadata +
                              config_pre_post_processor.timeseries):
        if field in config_pre_post_processor.metadata:
            metric_class_name = "Single attribute distributional similarity"
            class_name = "AttrDistSimilarity"
        elif field in config_pre_post_processor.timeseries:
            metric_class_name = "Single feature distributional similarity"
            class_name = "FeatureDistSimilarity"
        if 'bit' in getattr(field, 'encoding', '') or \
            'word2vec' in getattr(field, 'encoding', '') or \
                'categorical' in getattr(field, 'encoding', ''):
            sdmetrics_config["metadata"]["fields"][
                field.column] = {
                "type": "categorical"}
        if getattr(field, 'type', '') == 'float':
            sdmetrics_config["metadata"]["fields"][
                field.column] = {
                "type": "numerical"}
        sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                metric_class_name: {
                    "class": class_name,
                    "target_list": [[field.column]],
                    "configs": {
                        "categorical_mapping": getattr(field, 'categorical_mapping', True),
                        "comparison_type": comparison_type
                    }
                }
            }
        )

    # Add session length metric if the dataset is a pcap
    if config_pre_post_processor.dataset_type == 'pcap':
        sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                "Session length distributional similarity": {
                    "class": "SessionLengthDistSimilarity",
                    "configs": {
                        "comparison_type": comparison_type
                    }
                }
            }
        )
    if config_pre_post_processor.timestamp.generation:
        sdmetrics_config["metadata"]["fields"][
            config_pre_post_processor.timestamp.column] = {
            "type": "numerical"}
        sdmetrics_config["config"]["metrics"]["fidelity"].append(
            {
                "Single feature distributional similarity": {
                    "class": "FeatureDistSimilarity",
                    "target_list": [
                        [
                            config_pre_post_processor.timestamp.column
                        ]
                    ],
                    "configs": {
                        "comparison_type": comparison_type
                    }
                }
            }
        )
    sdmetrics_config["metadata"]["entity_columns"] = [
        field.column for field in config_pre_post_processor.metadata
    ]
    sdmetrics_config["metadata"]["sequence_index"] = config_pre_post_processor.timestamp.column if config_pre_post_processor.timestamp.generation else None
    sdmetrics_config["metadata"]["context_columns"] = []

    return sdmetrics_config


def _last_lvl_folder(folder):
    return str(Path(folder).parents[0])


def IP_int2str(IP_int):
    return str(ipaddress.ip_address(IP_int))


def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))


def IPs_int2str(IPs_int):
    return [IP_int2str(i) for i in IPs_int]


def IPs_str2int(IPs_str):
    return [IP_str2int(i) for i in IPs_str]


pr_dict = {
    "ESP": 50,
    "GRE": 47,
    "ICMP": 1,
    "IPIP": 4,
    "IPv6": 41,
    "TCP": 6,
    "UDP": 17,
    "Other": 255
}


def prs_str2int(prs):
    prs_int = []
    for p in prs:
        prs_int.append(pr_dict[p])
    return prs_int


pr_int2str_dict = {
    1: "ICMP",
    4: "IPIP",
    6: "TCP",
    17: "UDP",
    41: "IPv6",
    47: "GRE",
    50: "ESP",
    255: "Other"
}


def prs_int2str(prs_int):
    prs_str = []
    for p in prs_int:
        prs_str.append(pr_int2str_dict[p])
    return prs_str


def csv2pcap_single(input, output):
    # df = pd.read_csv(input).sort_values(["time"])
    df = input.sort_values(["time"])

    packets = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        time = float(row["time"] / 10**6)
        if isinstance(row["srcip"], str):
            srcip = IP_str2int(row["srcip"])
            dstip = IP_str2int(row["dstip"])
            src = socket.inet_ntoa(struct.pack('!L', srcip))
            dst = socket.inet_ntoa(struct.pack('!L', dstip))
        else:
            src = socket.inet_ntoa(struct.pack('!L', row["srcip"]))
            dst = socket.inet_ntoa(struct.pack('!L', row["dstip"]))

        srcport = row["srcport"]
        dstport = row["dstport"]
        proto = row["proto"]
        pkt_len = int(row["pkt_len"])

        try:
            proto = int(proto)
        except BaseException:
            if proto == "TCP":
                proto = 6
            elif proto == "UDP":
                proto = 17
            elif proto == "ICMP":
                proto = 1
            else:
                proto = 0

        ip = IP(src=src, dst=dst, len=pkt_len, proto=proto)
        if proto == 1:
            p = ip / ICMP()
        elif proto == 6:
            tcp = TCP(sport=srcport, dport=dstport)
            p = ip / tcp
        elif proto == 17:
            udp = UDP(sport=srcport, dport=dstport)
            p = ip / udp
        else:
            p = ip

        p.time = time
        p.len = pkt_len
        p.wirelen = pkt_len + 4

        packets.append(p)

    wrpcap(output, packets)
