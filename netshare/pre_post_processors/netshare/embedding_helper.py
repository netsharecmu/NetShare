import itertools
import numpy as np

from annoy import AnnoyIndex
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def build_annoy_dictionary_word2vec(
        df,
        model_path,
        word2vec_cols,
        word2vec_size,
        n_trees):

    model = Word2Vec.load(model_path)
    wv = model.wv

    # type : [cols]
    # ("ip": ["srcip", "dstip"])
    # "port": ["srcport", "dstport"]
    # "proto": ["proto"]
    dict_type_cols = {}
    for col in word2vec_cols:
        type = col.encoding.split("_")[1]
        if type not in dict_type_cols:
            dict_type_cols[type] = []
        dict_type_cols[type].append(col.column)
    print(dict_type_cols)

    sets = []
    dict_type_annDictPair = {}
    for type, cols in dict_type_cols.items():
        type_set = set(list(itertools.chain.from_iterable(
            [list(df[col]) for col in cols])))
        type_ann = AnnoyIndex(word2vec_size, 'angular')
        type_dict = {}
        index = 0

        for ele in type_set:
            type_ann.add_item(index, get_vector(
                model, str(ele), norm_option=True))
            type_dict[index] = ele
            index += 1
        type_ann.build(n_trees)

        dict_type_annDictPair[type] = (type_ann, type_dict)

    print("Finish building Angular trees...")

    return dict_type_annDictPair


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
        print(f"{word} not in dict")
        print("Help!!!!")
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
