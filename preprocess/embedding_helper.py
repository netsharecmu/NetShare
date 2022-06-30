from annoy import AnnoyIndex
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
# from bytewise_embedding import predict, load_model
import numpy as np

from tqdm import tqdm

def build_annoy_dictionary_word2vec(csv, model, length, file_type="PCAP", n_trees=100, encode_IP=False):
    print("n_trees:", n_trees)

    if encode_IP == True:
        ip_set = set(list(set(csv["srcip"])) + list(set(csv["dstip"])))
    port_set = set(list(set(csv["srcport"])) + list(set(csv["dstport"])))
    proto_set = set(csv["proto"])
    
    if encode_IP == True:
        ann_ip = AnnoyIndex(length, 'angular')
    ann_port = AnnoyIndex(length, 'angular')
    ann_proto = AnnoyIndex(length, 'angular')

    model = Word2Vec.load(model)
    wv = model.wv

    if encode_IP == True:
        ip_dic = {}
        ip_index = 0

    port_dic = {}
    port_index = 0
    proto_dic = {}
    proto_index = 0

    if encode_IP == True:
        for ip in ip_set:
            ann_ip.add_item(ip_index, get_vector(model, str(ip), norm_option=True))
            # ann_ip.add_item(ip_index, wv[str(ip)])
            ip_dic[ip_index] = ip
            ip_index += 1
    
    for port in port_set:
        ann_port.add_item(port_index, get_vector(model, str(port), norm_option=True))
        # ann_port.add_item(port_index, wv[str(port)]) # every ip/port/proto should be in the ``wv'' as this is used to construct the model.
        port_dic[port_index] = port
        port_index += 1
    
    for proto in proto_set:
        ann_proto.add_item(proto_index, get_vector(model, str(proto), norm_option=True))
        # ann_proto.add_item(proto_index, wv[str(proto)])
        proto_dic[proto_index] = proto
        proto_index += 1
            
    if encode_IP == True:
        ann_ip.build(n_trees)
    ann_port.build(n_trees)
    ann_proto.build(n_trees)

    
    if file_type == "PCAP" or file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
        if encode_IP == True:
            return (ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic)
        else:
            return (ann_port, port_dic, ann_proto, proto_dic)
    else:
        raise ValueError("Unsupported file type!")

# def build_annoy_dictionary_bytewise(csv, model, embedding_size):
#     ip_set = set()
#     ann_ip = AnnoyIndex(embedding_size, 'angular')
#     model = load_model(model, embedding_size)
#     dic = {}
#     ip_index = 0

#     for row in range(0, len(csv)):
#         if csv.at[row, 'srcip'] not in ip_set:
#             ip_set.add(csv.at[row, 'srcip'])
#         if csv.at[row, 'dstip'] not in ip_set:
#             ip_set.add(csv.at[row, 'dstip'])
#     ip_list = list(ip_set)
#     dic = {}
#     res = predict(model, ip_list)
#     for i in range(0, len(ip_list)):
#         ann_ip.add_item(i, res[i])
#         dic[i] = ip_list[i]
#     ann_ip.build(20)
#     return (ann_ip, dic)

def get_original_obj(ann, vector, dic):
    obj_list = ann.get_nns_by_vector(vector, 1, search_k=-1, include_distances=False)

    return dic[obj_list[0]]
    

def get_original_objs(ann, vectors, dic):
    res = []
    for vector in vectors:
        obj_list = ann.get_nns_by_vector(vector, 1, search_k=-1, include_distances=False)
        res.append(dic[obj_list[0]])
    return res

# return vector for the given word
def get_vector(model, word, norm_option=False):
    all_words_str = list(model.wv.vocab.keys())

    # Privacy-related
    # If word not in the vocabulary, replace with nearest neighbor
    # suppose that protocol is covered while very few port numbers are out of range
    if word not in all_words_str:
        all_words = []
        for ele in all_words_str:
            if ele.isdigit():
                all_words.append(int(ele))
        all_words = np.array(all_words).reshape((-1, 1))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(all_words)
        distances, indices = nbrs.kneighbors([[int(word)]])
        nearest_word = str(all_words[indices[0][0]][0])
        # print("nearest_word:", nearest_word)
        model.init_sims()
        return model.wv.word_vec(nearest_word, use_norm=norm_option)
    else:
        model.init_sims()
        return model.wv.word_vec(word, use_norm=norm_option)

# a.get_nns_by_item(i, n, search_k=-1, include_distances=False) returns the n closest items. During the query it will inspect up to search_k nodes which defaults to n_trees * n if not provided. search_k gives you a run-time tradeoff between better accuracy and speed. If you set include_distances to True, it will return a 2 element tuple with two lists in it: the second one containing all corresponding distances.
# a.get_nns_by_vector(v, n, search_k=-1, include_distances=False) same but query by vector v.

# import pandas as pd
# # ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec
# ann_ip, ip_dic = build_annoy_dictionary_bytewise(pd.read_csv("../../../traces/caida/equinix-nyc.dirA.20180315-125910.UTC.anon/equinix-nyc.dirA.20180315-125910.UTC.anon.sample.1M.csv"), "bytewise_embed.model", 5)
# print(get_original_obj(ann_ip, [[0, 0, 0, 0, 0]], ip_dic))
