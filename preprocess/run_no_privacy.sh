python3 word2vec_embedding.py --src_dir ../data/1M/ugr16 --word_vec_size 10 --file_type UGR16
python3 preprocess_by_type.py --src_dir ../data/1M/ugr16 --word2vec_vecSize 10 --file_type UGR16 --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --netflow_interarrival

python3 word2vec_embedding.py --src_dir ../data/1M/cidds --word_vec_size 10 --file_type CIDDS
python3 preprocess_by_type.py --src_dir ../data/1M/cidds --word2vec_vecSize 10 --file_type CIDDS --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --netflow_interarrival

python3 word2vec_embedding.py --src_dir ../data/1M/ton --word_vec_size 10 --file_type TON
python3 preprocess_by_type.py --src_dir ../data/1M/ton --word2vec_vecSize 10 --file_type TON --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --netflow_interarrival

make clean && make
./pcapParser ../data/1M/caida/raw.pcap ../data/1M/caida/raw.csv
python3 word2vec_embedding.py --src_dir ../data/1M/caida --word_vec_size 10 --file_type PCAP
python3 preprocess_by_type.py --src_dir ../data/1M/caida --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header

make clean && make
./pcapParser ../data/1M/dc/raw.pcap ../data/1M/dc/raw.csv
python3 word2vec_embedding.py --src_dir ../data/1M/dc --word_vec_size 10 --file_type PCAP
python3 preprocess_by_type.py --src_dir ../data/1M/dc --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header

make clean && make
./pcapParser ../data/1M/ca/raw.pcap ../data/1M/ca/raw.csv
python3 word2vec_embedding.py --src_dir ../data/1M/ca --word_vec_size 10 --file_type PCAP
python3 preprocess_by_type.py --src_dir ../data/1M/ca --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header