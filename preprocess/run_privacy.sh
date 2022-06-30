# privacy-related: private data
# caida
python3 preprocess_by_type_privacy.py --src_dir ../data/1M_privacy/caida --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header --start_time 1521118773200000.0 --end_time 1521118775700000.0 --global_max_flow_len 5000 --min_interarrival_within_flow 0.0 --max_interarrival_within_flow 1.0

python3 preprocess_by_type_privacy.py --src_dir ../data/public/caida --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header --partial_epoch --partial_epoch_until 1 --start_time 1432213260000000 --end_time 1432213263400660 --global_max_flow_len 5000 --min_interarrival_within_flow 0.0 --max_interarrival_within_flow 1.0

python3 preprocess_by_type_privacy.py --src_dir ../data/public/dc --word2vec_vecSize 10 --file_type PCAP --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --pcap_interarrival --full_IP_header --partial_epoch --partial_epoch_until 1 --start_time 1261070332163486 --end_time 1261070605315021 --global_max_flow_len 5000 --min_interarrival_within_flow 0.0 --max_interarrival_within_flow 1.0


# ugr16
python3 preprocess_by_type_privacy.py --src_dir ../data/1M_privacy/ugr16 --word2vec_vecSize 10 --file_type UGR16 --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --netflow_interarrival --start_time 1458298020000000.0 --end_time 1458299460000000.0 --global_max_flow_len 100 --min_interarrival_within_flow 0.0 --max_interarrival_within_flow 300.0

python3 preprocess_by_type_privacy.py --src_dir ../data/public/ugr16 --word2vec_vecSize 10 --file_type UGR16 --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --netflow_interarrival --partial_epoch --partial_epoch_until 1 --start_time 1459962987028000.0 --end_time 1459963760740000.0 --global_max_flow_len 100 --min_interarrival_within_flow 0.0 --max_interarrival_within_flow 300.0

python3 preprocess_by_type_privacy.py --src_dir ../data/public/cidds --word2vec_vecSize 10 --file_type CIDDS --split_name multiepoch_dep_v2 --df2epochs fixed_time --n_instances 10 --netflow_interarrival --partial_epoch --partial_epoch_until 1 --start_time 1489536076551000.0 --end_time 1489772072840000.0 --global_max_flow_len 100 --min_interarrival_within_flow 0.0 --max_interarrival_within_flow 300.0