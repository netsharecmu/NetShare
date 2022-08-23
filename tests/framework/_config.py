from netshare.utils import ContinuousField, DiscreteField, BitField

config = {
    'original_data_file': 'traces/1M/caida/raw.pcap',
    # 'original_data_file': 'traces/1M/ugr16/raw.csv',
    # 'original_data_folder': 'traces/1M/caida',
    # 'file_extension': '.pcap',

    # Whether it is allowed to overwrite the results
    'overwrite': True,

    'pre_post_processor': {
        'class': 'NetsharePrePostProcessor',
        'config': {
            'dataset_type': 'pcap',
            # 'dataset_type': 'netflow',
            # 'fields': {
            #     "srcip": BitField
            # },
            'word2vec_vecSize': 10,

            # 0 for [0, 1] normalization, 1 for [-1, 1] normalization
            'norm_option': 0,

            # multiepoch_ind: multiple epochs, no addi metadata
            # multiepoch_dep_v1: cross-epoch, <five tuples, 1, 1, 0, 0, ... 1>
            #                                 <five tuples, 0, 0, 0, 0, ... 0>
            # multiepoch_dep_v2: cross-epoch, <five tuples, 1, 1, 0, 0, ... 1>
            #                                 <five tuples, 0, 1, 0, 0, ... 1>
            'split_name': 'multichunk_dep_v2',

            'df2chunks': 'fixed_time',  # fixed_time, fixed_size
            'n_chunks': 10,  # of number of chunks
            'timestamp': 'interarrival',  # timestamp encode: raw/interarrival

            # False: <five tuples, timestamp, pkt_len>
            # True: full IP header + port number from tcp/udp header
            'full_IP_header': True,

            'encode_IP': 'bit',  # 'bit' or 'word2vec' encoding for IP address
            'dp': False  # whether DP is required for the workflow
        }
    },
    'model_manager': {
        'class': 'NetShareManager',
        'config': {}
    },
    'model': {
        'class': 'DummyModel',
        'config': {}
    }
}
