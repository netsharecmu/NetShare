# Configuration files
We provided a list of [default configuration files](./default/) following the best practice we obtained from a variety of datasets. 

Here are a list of key parameters divided by different modules:

1. global_config: configurations shared by the following modules (i.e., pre_post_processor, model_manager, model)

|       Parameter      |                                    Description                                   |
|:--------------------:|:--------------------------------------------------------------------------------:|
|      `overwrite`     |                    True if overwritng `work_folder` is allowed                   |
| `original_data_file` |                               path to the raw data                               |
|    `dataset_type`    |                      Currently support "pcap" and "netflow"                      |
|      `n_chunks`      |                      Number of chunks to divide the trace.                       |
|         `dp`         |              True if differential privacy is expected. False if not.             |
|  `word2vec_vecSize`  |      The size of embedded "word" for word2vec model (encoding port/protocol)     |
|      `timestamp`     |       How to encode timestamp. Currently support "raw" and "interarrival".       |
|      `truncate`      | Truncate packets with timestamp out of real data's range. "per_chunk" or "none". |

2. [pre_post_processor](../pre_post_processors/): preprocess data from raw format (e.g., PCAP/NetFlow) to NetShare-compatabile; and postprocess generated data to original format (e.g., PCAP/NetFlow).


3. [model_manager](../model_managers/): manage training and generation of the model (e.g., how to use the first chunk as the seed chunk)


4. [model](../models/): SOTA time-series GAN model [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)
