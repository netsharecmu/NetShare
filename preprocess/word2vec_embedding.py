import os, argparse

from gensim.models import Word2Vec
import pandas as pd
import numpy as np

from embedding_helper import build_annoy_dictionary_word2vec, get_original_obj, get_vector
from sklearn.neighbors import NearestNeighbors


def preprocess(csv, file_type="PCAP", encode_IP=False):
    sentences = []
    for row in range(0, len(csv)):
        if file_type == "PCAP" or file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
            if encode_IP == True:
                sentence = [csv.at[row, 'srcip'], csv.at[row, 'dstip'], csv.at[row, 'srcport'], csv.at[row, 'dstport'], csv.at[row, 'proto']]
            else:
                sentence = [csv.at[row, 'srcport'], csv.at[row, 'dstport'], csv.at[row, 'proto']]
    

        elif file_type == "FBFLOW":
            sentence = [csv.at[row, 'srcip'], csv.at[row, 'dstip'], csv.at[row, 'srcport'], csv.at[row, 'dstport'], csv.at[row, 'proto'], csv.at[row, 'srchostprefix'], csv.at[row, 'dsthostprefix'], csv.at[row, 'srcrack'], csv.at[row, 'dstrack'], csv.at[row, 'srcpod'], csv.at[row, 'dstpod']]

        sentence = list(map(str, sentence))
        sentences.append(sentence)

    return sentences

# # return vector for the given word
# def get_vector(model, word, norm_option=False):
#     all_words_str = list(model.wv.vocab.keys())

#     # Privacy-related
#     # If word not in the vocabulary, replace with nearest neighbor
#     # suppose that protocol is covered while very few port numbers are out of range
#     if word not in all_words_str:
#         all_words = []
#         for ele in all_words_str:
#             if ele.isdigit():
#                 all_words.append(int(ele))
#         all_words = np.array(all_words).reshape((-1, 1))
#         nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(all_words)
#         distances, indices = nbrs.kneighbors([[int(word)]])
#         nearest_word = str(all_words[indices[0][0]][0])
#         # print("nearest_word:", nearest_word)
#         model.init_sims()
#         return model.wv.word_vec(nearest_word, use_norm=norm_option)
#     else:
#         model.init_sims()
#         return model.wv.word_vec(word, use_norm=norm_option)

# return word for the given vector
# def get_word(model, vec):
#     return model.wv.most_similar(positive=[vec], topn=1)[0][0]

def test_embed_bidirectional(model_file, ann, dic, word):
    model = Word2Vec.load(model_file)

    raw_vec = get_vector(model, word, False)
    normed_vec = get_vector(model, word, True)

    print("word: {}, vector(raw): {}".format(word, raw_vec))
    print("word: {}, vector(l2-norm): {}".format(word, normed_vec))

    print("vec(raw): {}, word: {}".format(raw_vec, get_original_obj(ann, raw_vec, dic)))
    print("vec(l2-norm): {}, word: {}".format(normed_vec, get_original_obj(ann, normed_vec, dic)))
    print()

def test_model(df, model, vec_len, file_type, n_trees, encode_IP=False):
    if file_type == "PCAP" or file_type == "UGR16" or file_type == "CIDDS" or file_type == "TON":
        if encode_IP == True:
            ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(df, model, vec_len, file_type=file_type, n_trees=n_trees, encode_IP=encode_IP)
        else:
            ann_port, port_dic, ann_proto, proto_dic = build_annoy_dictionary_word2vec(df, model, vec_len, file_type=file_type, n_trees=n_trees, encode_IP=encode_IP)

    elif file_type == "FBFLOW":
        ann_ip, ip_dic, ann_port, port_dic, ann_proto, proto_dic, ann_hostprefix, hostprefix_dic, ann_rack, rack_dic, ann_pod, pod_dic = build_annoy_dictionary_word2vec(df, model, vec_len, file_type=file_type, n_trees=n_trees, encode_IP=encode_IP)
    
    if encode_IP == True:
        ip_word = str(df.at[10, 'srcip'])
    # port_word = str(df.at[10, 'srcport'])
    port_word = "443"
    proto_word = str(df.at[10, 'proto'])

    if file_type == "FBFLOW":
        hostprefix_word = str(df.at[10, 'srchostprefix'])
        rack_word = str(df.at[10, 'srcrack'])
        pod_word = str(df.at[10, 'srcpod'])

    if encode_IP == True:
        test_embed_bidirectional(model, ann_ip, ip_dic, ip_word)
    test_embed_bidirectional(model, ann_port, port_dic, port_word)
    test_embed_bidirectional(model, ann_proto, proto_dic, proto_word)

    if file_type == "FBFLOW":
        test_embed_bidirectional(model, ann_hostprefix, hostprefix_dic, hostprefix_word)
        test_embed_bidirectional(model, ann_rack, rack_dic, rack_word)
        test_embed_bidirectional(model, ann_pod, pod_dic, pod_word)

def main(args):
    model_name = os.path.join(args.src_dir, "word2vec_vecSize_{}.model".format(args.word_vec_size))
    df = pd.read_csv(os.path.join(args.src_dir, args.src_csv))

    if os.path.exists(model_name):
        print("loading pre-trained model...")
        model = Word2Vec.load(model_name)

    else:
        print("Training from scratch...")
        sentences = preprocess(df, args.file_type)
        if args.file_type == "PCAP" or args.file_type == "UGR16" or args.file_type == "CIDDS" or args.file_type == "TON":
            model = Word2Vec(sentences=sentences, size=args.word_vec_size, window=5, min_count=1, workers=10)
        elif args.file_type == "FBFLOW":
            model = Word2Vec(sentences=sentences, size=args.word_vec_size, window=10, min_count=1, workers=10)

        model.save(model_name)
    
    if args.model_test:
        test_model(df, model_name, args.word_vec_size, file_type=args.file_type, n_trees=args.n_trees, encode_IP=args.encode_IP)



# vector = model.wv['6']
# print(model.wv['6'], model.wv['17'])

# word = model.most_similar(positive=[vector], topn=1)
# print(word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', type=str, default="../traces/caida_sliced_test/equinix-nyc.dirA.20180719-125910.UTC.anon_slice3")
    parser.add_argument('--src_csv', type=str, default="raw.csv")
    parser.add_argument('--word_vec_size', type=int, default=32)

    # used for annoy package
    parser.add_argument('--n_trees', type=int, default=100)

    # PCAP, UGR16, CIDDS, FBFLOW, VALTIX
    # PCAP: CAIDA, Data Center, CA
    parser.add_argument('--file_type', type=str, default="PCAP")
    parser.add_argument('--encode_IP', action='store_true', default=False)

    parser.add_argument('--model_test', action='store_true', default=False)
    

    args = parser.parse_args()

    main(args) 