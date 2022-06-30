from annoy import AnnoyIndex
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
# from bytewise_embedding import predict, load_model
import numpy as np

from tqdm import tqdm

def build_annoy_dictionary_word2vec(csv, model, length, file_type="PCAP", n_trees=100, encode_IP=False):
    print("n_trees:", n_trees)

    if encode_IP == True:
        ip_set = set()
    port_set = set()
    proto_set = set()
    if file_type == "FBFLOW":
        hostprefix_set = set()
        rack_set = set()
        pod_set = set()
    
    if encode_IP == True:
        ann_ip = AnnoyIndex(length, 'angular')
    ann_port = AnnoyIndex(length, 'angular')
    ann_proto = AnnoyIndex(length, 'angular')
    if file_type == "FBFLOW":
        ann_hostprefix = AnnoyIndex(length, 'angular')
        ann_rack = AnnoyIndex(length, 'angular')
        ann_pod = AnnoyIndex(length, 'angular')

    model = Word2Vec.load(model)
    wv = model.wv

    if encode_IP == True:
        ip_dic = {}
        ip_index = 0

    port_dic = {}
    port_index = 0
    proto_dic = {}
    proto_index = 0
    if file_type == "FBFLOW":
        hostprefix_dic = {}
        hostprefix_index = 0
        rack_dic = {}
        rack_index = 0
        pod_dic = {}
        pod_index = 0

    for row in tqdm(range(0, len(csv))):
        if encode_IP == True:
            if csv.at[row, 'srcip'] not in ip_set:
                ip_set.add(csv.at[row, 'srcip'])
                ann_ip.add_item(ip_index, wv[str(csv.at[row, 'srcip'])])
                ip_dic[ip_index] = csv.at[row, 'srcip']
                ip_index += 1
            if csv.at[row, 'dstip'] not in ip_set:
                ip_set.add(csv.at[row, 'dstip'])
                ann_ip.add_item(ip_index, wv[str(csv.at[row, 'dstip'])])
                ip_dic[ip_index] = csv.at[row, 'dstip']
                ip_index += 1
        
        if csv.at[row, 'srcport'] not in port_set:
            port_set.add(csv.at[row, 'srcport'])
            # ann_port.add_item(port_index, wv[str(csv.at[row, 'srcport'])])
            ann_port.add_item(port_index, get_vector(model, str(csv.at[row, 'srcport']), norm_option=True))
            port_dic[port_index] = csv.at[row, 'srcport']
            port_index += 1
        if csv.at[row, 'dstport'] not in port_set:
            port_set.add(csv.at[row, 'dstport'])
            # ann_port.add_item(port_index, wv[str(csv.at[row, 'dstport'])])
            ann_port.add_item(port_index, get_vector(model, str(csv.at[row, 'dstport']), norm_option=True))
            port_dic[port_index] = csv.at[row, 'dstport']
            port_index += 1
        if csv.at[row, 'proto'] not in proto_set:
            proto_set.add(csv.at[row, 'proto'])
            # ann_proto.add_item(proto_index, wv[str(csv.at[row, 'proto'])])
            ann_proto.add_item(proto_index, get_vector(model, str(csv.at[row, 'proto']), norm_option=True))
            proto_dic[proto_index] = csv.at[row, 'proto']
            proto_index += 1
        
        if file_type == "FBFLOW":
            if csv.at[row, 'srchostprefix'] not in hostprefix_set:
                hostprefix_set.add(csv.at[row, 'srchostprefix'])
                ann_hostprefix.add_item(hostprefix_index, wv[str(csv.at[row, 'srchostprefix'])])
                hostprefix_dic[hostprefix_index] = csv.at[row, 'srchostprefix']
                hostprefix_index += 1
            if csv.at[row, 'dsthostprefix'] not in hostprefix_set:
                hostprefix_set.add(csv.at[row, 'dsthostprefix'])
                ann_hostprefix.add_item(hostprefix_index, wv[str(csv.at[row, 'dsthostprefix'])])
                hostprefix_dic[hostprefix_index] = csv.at[row, 'dsthostprefix']
                hostprefix_index += 1
            if csv.at[row, 'srcrack'] not in rack_set:
                rack_set.add(csv.at[row, 'srcrack'])
                ann_rack.add_item(rack_index, wv[str(csv.at[row, 'srcrack'])])
                rack_dic[rack_index] = csv.at[row, 'srcrack']
                rack_index += 1
            if csv.at[row, 'dstrack'] not in rack_set:
                rack_set.add(csv.at[row, 'dstrack'])
                ann_rack.add_item(rack_index, wv[str(csv.at[row, 'dstrack'])])
                rack_dic[rack_index] = csv.at[row, 'dstrack']
                rack_index += 1
            if csv.at[row, 'srcpod'] not in pod_set:
                pod_set.add(csv.at[row, 'srcpod'])
                ann_pod.add_item(pod_index, wv[str(csv.at[row, 'srcpod'])])
                pod_dic[pod_index] = csv.at[row, 'srcpod']
                pod_index += 1
            if csv.at[row, 'dstpod'] not in pod_set:
                pod_set.add(csv.at[row, 'dstpod'])
                ann_pod.add_item(pod_index, wv[str(csv.at[row, 'dstpod'])])
                pod_dic[pod_index] = csv.at[row, 'dstpod']
                pod_index += 1
            
    if encode_IP == True:
        ann_ip.build(n_trees)
    ann_port.build(n_trees)
    ann_proto.build(n_trees)
    if file_type == "FBFLOW":
        ann_hostprefix.build(n_trees)
        ann_rack.build(n_trees)
        ann_pod.build(n_trees)
    
    if file_type == "PCAP" or file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
        if encode_IP == True:
            return (ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic)
        else:
            return (ann_port, port_dic, ann_proto, proto_dic)
    elif file_type == "FBFLOW":
        return (ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic, ann_hostprefix, hostprefix_dic, ann_rack, rack_dic, ann_pod, pod_dic)

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
