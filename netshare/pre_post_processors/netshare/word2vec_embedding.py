import os
import random

from gensim.models import Word2Vec
import pandas as pd
import numpy as np

from .embedding_helper import build_annoy_dictionary_word2vec
from .embedding_helper import get_original_obj, get_vector
from sklearn.neighbors import NearestNeighbors


def test_embed_bidirectional(model_path, ann, dic, word):
    model = Word2Vec.load(model_path)

    raw_vec = get_vector(model, word, False)
    normed_vec = get_vector(model, word, True)

    print("word: {}, vector(raw): {}".format(word, raw_vec))
    print("word: {}, vector(l2-norm): {}".format(word, normed_vec))

    print("vec(raw): {}, word: {}".format(
        raw_vec, get_original_obj(ann, raw_vec, dic)))
    print("vec(l2-norm): {}, word: {}".format(normed_vec,
          get_original_obj(ann, normed_vec, dic)))
    print()


def test_model(
    df,
    model_path,
    word2vec_cols,
    word2vec_size,
    annoy_n_trees
):
    dict_type_annDictPair = build_annoy_dictionary_word2vec(
        df=df,
        model_path=model_path,
        word2vec_cols=word2vec_cols,
        word2vec_size=word2vec_size,
        n_trees=annoy_n_trees
    )

    for col in word2vec_cols:
        type = col.encoding.split("_")[1]
        word = random.choice(df[col.column])
        print("Testing {col.column}...")
        test_embed_bidirectional(
            model_path=model_path,
            ann=dict_type_annDictPair[type][0],
            dic=dict_type_annDictPair[type][1],
            word=word)


def word2vec_train(
    df,
    out_dir,
    model_name,
    word2vec_cols,
    word2vec_size,
    annoy_n_trees,
    force_retrain=False,  # retrain from scratch
    model_test=False
):
    model_path = os.path.join(
        out_dir,
        "{}_{}.model".format(model_name, word2vec_size))

    if os.path.exists(model_path) and not force_retrain:
        print("Loading Word2Vec pre-trained model...")
        model = Word2Vec.load(model_path)
    else:
        print("Training Word2Vec model from scratch...")
        sentences = []
        for row in range(0, len(df)):
            sentence = [str(df.at[row, col])
                        for col in [c.column for c in word2vec_cols]]
            sentences.append(sentence)

        model = Word2Vec(
            sentences=sentences,
            size=word2vec_size,
            window=5,
            min_count=1,
            workers=10)
        model.save(model_path)
    print(f"Word2Vec model is saved at {model_path}")

    if model_test:
        test_model(
            df=df,
            model_path=model_path,
            word2vec_cols=word2vec_cols,
            word2vec_size=word2vec_size,
            annoy_n_trees=annoy_n_trees
        )

    return model_path
