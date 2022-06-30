'''EXPERIMENTAL CLEANUP FOR CA DATASET'''
'''PATH NAME HARDCODED, NOT FOR FUTURE USE'''

import os, sys, copy, pickle
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from gan import output
sys.modules["output"] = output

from gan import field
sys.modules["field"] = field

sys.path.append("../preprocess")
from embedding_helper import build_annoy_dictionary_word2vec, get_original_obj

# change this to true for (1) PCAP (2) uses interarrival
# PCAP_INTERARRIVAL=False
# WORD2VEC_SIZE=5


# NUM_CHUNKS=10

# if PCAP_INTERARRIVAL == False:
#     bit_idx_flagstart=224
# else:
#     bit_idx_flagstart=225


def last_lvl_folder(folder):
    return str(Path(folder).parents[0])

# [chunk_0_npz, chunk_1_npz, ...]
def merge_npzs(attr_raw_npz_folder, WORD2VEC_SIZE, PCAP_INTERARRIVAL, NUM_CHUNKS):
    # 128 is IP-bit size
    if PCAP_INTERARRIVAL == 0:
        bit_idx_flagstart=128+WORD2VEC_SIZE*3
    else:
        bit_idx_flagstart=128+WORD2VEC_SIZE*3+1

    print("PCAP_INTERARRIVAL:", PCAP_INTERARRIVAL)
    print("bit_idx_flagstart:", bit_idx_flagstart)

    attr_clean_npz_folder = os.path.join(str(Path(attr_raw_npz_folder).parents[0]), "attr_clean")
    os.makedirs(attr_clean_npz_folder, exist_ok=True)

    dict_chunkid_attr = {}
    dict_chunkid_attrset = {} # for cleanup purpose
    for chunkid in tqdm(range(NUM_CHUNKS)):
        dict_chunkid_attr[chunkid] = []
        dict_chunkid_attrset[chunkid] = set()

    for chunkid in tqdm(range(NUM_CHUNKS)):
        # HARDCODED, NOT INTENDED FOR FUTURE USE
        DATA_DIR_PERCHUNK = "../data/1M/ca/split-multiepoch_dep_v2,epochid-{},maxFlowLen-20376,Norm-ZERO_ONE,vecSize-10,df2epochs-fixed_time,interarrival-True,fullIPHdr-True,encodeIP-False/".format(chunkid)

        fin = open(os.path.join(DATA_DIR_PERCHUNK, "fields.pkl"), "rb")
        fields = pickle.load(fin)
        fin.close()

        word2vec_model_path = os.path.join(last_lvl_folder(DATA_DIR_PERCHUNK), "word2vec_vecSize_{}.model".format(WORD2VEC_SIZE))
        per_chunk_raw_df = pd.read_csv(os.path.join(DATA_DIR_PERCHUNK, "raw.csv"))
        ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(per_chunk_raw_df, word2vec_model_path, WORD2VEC_SIZE, file_type="PCAP", n_trees=1000, encode_IP=False)

        n_flows_startFromThisEpoch = 0
        n_flows_duplicated_crossEpoch = 0

        if not os.path.exists(os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid))):
            print("{} not exists...".format(os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid))))
            continue

        raw_attr_chunk = np.load(os.path.join(attr_raw_npz_folder, "chunk_id-{}.npz".format(chunkid)))["data_attribute"]

        for row in tqdm(raw_attr_chunk):
            # if row[bit_idx_flagstart] < row[bit_idx_flagstart+1]:
            if row[bit_idx_flagstart] < row[bit_idx_flagstart+1] and row[bit_idx_flagstart+2*chunkid+2] < row[bit_idx_flagstart+2*chunkid+3]:
                # denormalize five tuples
                srcip = list(row[0:64])
                dstip = list(row[64:128])
                srcport = list(row[128:128+WORD2VEC_SIZE])
                dstport = list(row[128+WORD2VEC_SIZE:128+WORD2VEC_SIZE*2])
                proto = list(row[128+WORD2VEC_SIZE*2:128+WORD2VEC_SIZE*3])

                srcip = fields["srcip"].denormalize(srcip)
                dstip = fields["dstip"].denormalize(dstip)
                srcport = get_original_obj(ann_port, srcport, port_dic)
                dstport = get_original_obj(ann_port, dstport, port_dic)
                proto = get_original_obj(ann_proto, proto, proto_dic)

                if (srcip, dstip, srcport, dstport, proto) in dict_chunkid_attrset[chunkid]:
                    n_flows_duplicated_crossEpoch += 1
                    continue

                # this chunk
                row_this_chunk = list(copy.deepcopy(row)[:bit_idx_flagstart])
                row_this_chunk += [0.0, 1.0]
                row_this_chunk += [1.0, 0.0]*(chunkid+1)
                for i in range(chunkid+1, NUM_CHUNKS):
                    if row[bit_idx_flagstart+2*i+2] < row[bit_idx_flagstart+2*i+3]:
                        row_this_chunk += [0.0, 1.0]
                    else:
                        row_this_chunk += [1.0, 0.0]
                # dict_chunkid_attr[chunkid].append(row_this_chunk)
                dict_chunkid_attr[chunkid].append(row)

                # following chunks
                # row_following_chunk = list(copy.deepcopy(row)[:bit_idx_flagstart])
                # row_following_chunk += [1.0, 0.0]*(1+NUM_CHUNKS)
                n_flows_startFromThisEpoch += 1
                row_following_chunk = list(copy.deepcopy(row))
                row_following_chunk[bit_idx_flagstart] = 1.0
                row_following_chunk[bit_idx_flagstart+1] = 0.0

                for i in range(chunkid+1, NUM_CHUNKS):
                    if row[bit_idx_flagstart+2*i+2] < row[bit_idx_flagstart+2*i+3]:
                        dict_chunkid_attr[i].append(row_following_chunk)
                        # dict_chunkid_attr[i].append(row)
                        dict_chunkid_attrset[i].add((srcip, dstip, srcport, dstport, proto))

        print("n_flows_startFromThisEpoch / total flows: {}/{}".format(n_flows_startFromThisEpoch, raw_attr_chunk.shape[0]))
        print("n_flows_duplicated_crossEpoch / total flows: {}/{}".format(n_flows_duplicated_crossEpoch, raw_attr_chunk.shape[0]))

        print("chunkid_attrset status...")
        for chunkid, attr_set in dict_chunkid_attrset.items():
            print("chunkid: {}, # of attrs: {}".format(chunkid, len(attr_set)))
    
    print("Saving merged attrs...")
    n_merged_attrs = 0
    for chunkid, attr_clean in dict_chunkid_attr.items():
        print("chunk {}: {} flows".format(chunkid, len(attr_clean)))
        n_merged_attrs += len(attr_clean)
        np.savez(os.path.join(attr_clean_npz_folder, "chunk_id-{}.npz".format(chunkid)), data_attribute=np.asarray(attr_clean))
    
    print("n_merged_attrs:", n_merged_attrs)

attr_raw_npz_folder = sys.argv[1]
WORD2VEC_SIZE = int(sys.argv[2])
PCAP_INTERARRIVAL = int(sys.argv[3])
NUM_CHUNKS = int(sys.argv[4])

merge_npzs(attr_raw_npz_folder, WORD2VEC_SIZE, PCAP_INTERARRIVAL, NUM_CHUNKS)

# merge_npzs("../results_eval/sigcomm2022/ugr16,Norm-MINUSONE_ONE/attr_raw")


