import itertools
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors

from netshare.utils.logger import logger
from netshare.utils.paths import (
    get_annoy_dict_idx_ele_for_word2vec,
    get_annoyIndex_for_word2vec,
    get_word2vec_model_path,
)


def word2vec_train(
    df: pd.DataFrame,
    model_name: str,
    word2vec_cols: List[Any],
    word2vec_size: int,
    annoy_n_trees: int,
    force_retrain: bool = False,  # retrain from scratch
    model_test: bool = False,
) -> None:
    model_path = get_word2vec_model_path()

    if os.path.exists(model_path) and not force_retrain:
        logger.info("Loading Word2Vec pre-trained model...")
        model = Word2Vec.load(model_path)
    else:
        logger.info("Training Word2Vec model from scratch...")
        sentences = []
        for row in range(0, len(df)):
            # TODO: If some lines were dropped (e.g. due to NaN), the rows will not be continuous
            sentence = [
                str(df.at[row, col]) for col in [c.column for c in word2vec_cols]
            ]
            sentences.append(sentence)

        model = Word2Vec(
            sentences=sentences, size=word2vec_size, window=5, min_count=1, workers=10
        )
        model.save(model_path)
    logger.info(f"Word2Vec model is saved at {model_path}")

    return None


def get_word2vec_type_col(word2vec_cols: List[Any]) -> Dict[str, List[str]]:
    dict_type_cols: Dict[str, List[str]] = defaultdict(list)
    for col in word2vec_cols:
        dict_type_cols[col.encoding].append(col.column)
    return dict_type_cols


def build_annoy_dictionary_word2vec(
    df: pd.DataFrame,
    model_path: str,
    word2vec_cols: List[Any],
    word2vec_size: int,
    n_trees: int,
) -> None:

    dict_encoding_type_vs_idx_ele_dict_pair = {}

    model = Word2Vec.load(model_path)
    wv = model.wv

    # encoding type and column names may not be necessarily the same. eg. for a pcap/netflow dataset
    # encoding type = word2vec_port -> column: ["srcport", "dstport"]
    # encoding type = word2vec_proto -> column: ["proto"]
    # srcport and dstport are two fields for a dataset, but from a word2vec perspective, they are both ports
    dict_encoding_type_vs_cols = get_word2vec_type_col(word2vec_cols)
    logger.debug(f"word2vec type columns are: {dict_encoding_type_vs_cols}")

    for encoding_type, cols in dict_encoding_type_vs_cols.items():
        ele_set = set(itertools.chain.from_iterable([list(df[col]) for col in cols]))
        type_ann = AnnoyIndex(word2vec_size, "angular")
        dict_idx_ele = {}

        for index, ele in enumerate(ele_set):
            type_ann.add_item(index, get_vector(model, str(ele), norm_option=True))
            dict_idx_ele[index] = ele
        type_ann.build(n_trees)

        dict_encoding_type_vs_idx_ele_dict_pair[encoding_type] = dict_idx_ele

        type_ann.save(get_annoyIndex_for_word2vec(encoding_type))
        with open(get_annoy_dict_idx_ele_for_word2vec(), "w") as outfile:
            json.dump(dict_encoding_type_vs_idx_ele_dict_pair, outfile)

    logger.info("Finish building Angular trees...")

    return None


def get_original_obj(ann: AnnoyIndex, vector: np.ndarray, dic: Dict[int, Any]) -> Any:
    obj_list = ann.get_nns_by_vector(vector, 1, search_k=-1, include_distances=False)

    return dic[obj_list[0]]


def get_original_objs(
    ann: AnnoyIndex, vectors: np.ndarray, dic: Dict[int, Any]
) -> List[Any]:
    return [get_original_obj(ann, vector, dic) for vector in vectors]


def get_vector(model: Word2Vec, word: str, norm_option: bool = False) -> np.ndarray:
    all_words_str = list(model.wv.vocab.keys())
    if word not in all_words_str:
        all_words = []
        for ele in all_words_str:
            if ele.isdigit():
                all_words.append(int(ele))
        all_words = np.array(all_words).reshape((-1, 1))
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(all_words)
        distances, indices = nbrs.kneighbors([[int(word)]])
        nearest_word = str(all_words[indices[0][0]][0])
        model.init_sims()
        return model.wv.word_vec(nearest_word, use_norm=norm_option)
    else:
        model.init_sims()
        return model.wv.word_vec(word, use_norm=norm_option)
