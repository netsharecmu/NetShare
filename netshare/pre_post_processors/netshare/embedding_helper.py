import numpy as np

from annoy import AnnoyIndex
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def build_annoy_dictionary_word2vec(
        csv, model, length,
        file_type="pcap", n_trees=100, encode_IP='bit'):
    print("n_trees:", n_trees)

    if encode_IP == 'word2vec':
        ip_set = set(list(set(csv["srcip"])) + list(set(csv["dstip"])))
    port_set = set(list(set(csv["srcport"])) + list(set(csv["dstport"])))
    proto_set = set(csv["proto"])

    print("Finish building field set...")

    if encode_IP == 'word2vec':
        ann_ip = AnnoyIndex(length, 'angular')
    ann_port = AnnoyIndex(length, 'angular')
    ann_proto = AnnoyIndex(length, 'angular')

    model = Word2Vec.load(model)
    wv = model.wv

    if encode_IP == 'word2vec':
        ip_dic = {}
        ip_index = 0

    port_dic = {}
    port_index = 0
    proto_dic = {}
    proto_index = 0

    if encode_IP == 'word2vec':
        for ip in ip_set:
            ann_ip.add_item(ip_index, get_vector(
                model, str(ip), norm_option=True))
            # ann_ip.add_item(ip_index, wv[str(ip)])
            ip_dic[ip_index] = ip
            ip_index += 1

    for port in port_set:
        ann_port.add_item(port_index, get_vector(
            model, str(port), norm_option=True))
        # ann_port.add_item(port_index, wv[str(port)])
        # # every ip/port/proto should be in the ``wv''
        # as this is used to construct the model.
        port_dic[port_index] = port
        port_index += 1

    for proto in proto_set:
        ann_proto.add_item(proto_index, get_vector(
            model, str(proto), norm_option=True))
        # ann_proto.add_item(proto_index, wv[str(proto)])
        proto_dic[proto_index] = proto
        proto_index += 1

    if encode_IP == 'word2vec':
        ann_ip.build(n_trees)
    ann_port.build(n_trees)
    ann_proto.build(n_trees)

    print("Finish building Angular trees...")

    if encode_IP == 'word2vec':
        return (ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic)
    elif encode_IP == 'bit':
        return (ann_port, port_dic, ann_proto, proto_dic)


def get_original_obj(ann, vector, dic):
    obj_list = ann.get_nns_by_vector(
        vector, 1, search_k=-1, include_distances=False)

    return dic[obj_list[0]]


def get_original_objs(ann, vectors, dic):
    res = []
    for vector in vectors:
        obj_list = ann.get_nns_by_vector(
            vector, 1, search_k=-1, include_distances=False)
        res.append(dic[obj_list[0]])
    return res

# return vector for the given word


def get_vector(model, word, norm_option=False):
    all_words_str = list(model.wv.vocab.keys())

    # Privacy-related
    # If word not in the vocabulary, replace with nearest neighbor
    # Suppose that protocol is covered
    #   while very few port numbers are out of range
    if word not in all_words_str:
        # print(f"{word} not in dict")
        all_words = []
        for ele in all_words_str:
            if ele.isdigit():
                all_words.append(int(ele))
        all_words = np.array(all_words).reshape((-1, 1))
        nbrs = NearestNeighbors(
            n_neighbors=1, algorithm='ball_tree').fit(all_words)
        distances, indices = nbrs.kneighbors([[int(word)]])
        nearest_word = str(all_words[indices[0][0]][0])
        # print("nearest_word:", nearest_word)
        model.init_sims()
        return model.wv.word_vec(nearest_word, use_norm=norm_option)
    else:
        model.init_sims()
        return model.wv.word_vec(word, use_norm=norm_option)
