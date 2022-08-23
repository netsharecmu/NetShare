from base64 import encode
import os
import argparse

from gensim.models import Word2Vec
import pandas as pd
import numpy as np

from .embedding_helper import build_annoy_dictionary_word2vec
from .embedding_helper import get_original_obj, get_vector
from sklearn.neighbors import NearestNeighbors


def preprocess(csv, file_type="pcap", encode_IP='bit'):
    '''
    file_type: pcap, netflow
    encode_IP: bit, word2vec
    '''
    sentences = []
    for row in range(0, len(csv)):
        if encode_IP == 'word2vec':
            sentence = [csv.at[row, 'srcip'], csv.at[row, 'dstip'],
                        csv.at[row, 'srcport'], csv.at[row, 'dstport'],
                        csv.at[row, 'proto']]
        elif encode_IP == 'bit':
            sentence = [csv.at[row, 'srcport'],
                        csv.at[row, 'dstport'],
                        csv.at[row, 'proto']]

        sentence = list(map(str, sentence))
        sentences.append(sentence)

    return sentences


def test_embed_bidirectional(model_file, ann, dic, word):
    model = Word2Vec.load(model_file)

    raw_vec = get_vector(model, word, False)
    normed_vec = get_vector(model, word, True)

    print("word: {}, vector(raw): {}".format(word, raw_vec))
    print("word: {}, vector(l2-norm): {}".format(word, normed_vec))

    print("vec(raw): {}, word: {}".format(
        raw_vec, get_original_obj(ann, raw_vec, dic)))
    print("vec(l2-norm): {}, word: {}".format(normed_vec,
          get_original_obj(ann, normed_vec, dic)))
    print()


def test_model(df, model, vec_len, file_type, n_trees, encode_IP):
    if encode_IP == 'word2vec':
        ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic = \
            build_annoy_dictionary_word2vec(
                csv=df,
                model=model,
                length=vec_len,
                file_type=file_type,
                n_trees=n_trees,
                encode_IP=encode_IP)
    elif encode_IP == 'bit':
        ann_port, port_dic, ann_proto, proto_dic = \
            build_annoy_dictionary_word2vec(
                csv=df,
                model=model,
                length=vec_len,
                file_type=file_type,
                n_trees=n_trees,
                encode_IP=encode_IP)

    if encode_IP == 'word2vec':
        ip_word = str(df.at[10, 'srcip'])
    port_word = "443"
    proto_word = str(df.at[10, 'proto'])

    if encode_IP == 'word2vec':
        test_embed_bidirectional(model, ann_ip, ip_dic, ip_word)
    test_embed_bidirectional(model, ann_port, port_dic, port_word)
    test_embed_bidirectional(model, ann_proto, proto_dic, proto_word)


def word2vec_train(
    df,
    out_dir,
    file_type,
    word_vec_size=10,
    n_trees=100,
    encode_IP='bit',
    model_test=False
):
    model_name = os.path.join(
        out_dir, "word2vec_vecSize_{}.model".format(word_vec_size))

    if os.path.exists(model_name):
        print("Loading Word2Vec pre-trained model...")
        model = Word2Vec.load(model_name)
    else:
        print("Training Word2Vec model from scratch...")
        sentences = preprocess(
            csv=df,
            file_type=file_type,
            encode_IP=encode_IP)
        model = Word2Vec(
            sentences=sentences,
            size=word_vec_size,
            window=5,
            min_count=1,
            workers=10)
        model.save(model_name)
    print(f"Word2Vec model is saved at {model_name}")

    if model_test:
        test_model(
            df=df,
            model=model_name,
            vec_len=word_vec_size,
            file_type=file_type,
            n_trees=n_trees,
            encode_IP=encode_IP)

    return model_name
