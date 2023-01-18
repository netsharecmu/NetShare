# Configuration files
We provided a list of [default configuration files](./default/) following the best practice we obtained from a variety of datasets. For most cases, most parameters should be kept intact.

Here are a list of key parameters divided by different modules:

1. global_config: configurations shared by the following modules (i.e., pre_post_processor, model_manager, model)

|     Parameter      |                                        Description                                        |
|:------------------:|:-----------------------------------------------------------------------------------------:|
|    `overwrite`     |                        True if overwritng `work_folder` is allowed                        |
|    `ray_enabled`   |                    True if we should use ray for parallelism                              |
|     `n_chunks`     |                           Number of chunks to divide the trace.                           |
|        `dp`        |                  True if differential privacy is expected. False if not.                  |
|    `timestamp`     |           How to encode timestamp. Currently support "raw" and "interarrival".            |

2. input_adapters.data_source: Where to load the data from (local files, S3 bucket, etc.)

| Parameter |                                 Description                                 |
|:---------:|:---------------------------------------------------------------------------:|
|  `type`   | Which data source we should use (see `netshare.input_adapters.data_source`) |
|   `...`   |             Each data source define its own config in its class             |

3. input_adapters.format_normalizer: How to parse the given data to the canonical NetShare format (`.cap` files, csv, json. etc.)

| Parameter |                                             Description                                             |
|:---------:|:---------------------------------------------------------------------------------------------------:|
|  `type`   | Which format normalizer we should use (see `netshare.input_adapters.normalize_format_to_canonical`) |
|   `...`   |                      Each format normalizer define its own config in its class                      |

4. learn: How to train the model

|  Parameter  |                             Description                             |
|:-----------:|:-------------------------------------------------------------------:|
|   `num_train_samples`    |         How many samples we should use from the given data          |
|   `df2chunks`    | How to split the data to chunks. Options: fixed_time and fixed_size |
|   `split_name`    |           Options: multichunk_dep_v1 or multichunk_dep_v2           |
|   `max_flow_len`    |        The length of the longest session, or `null` if unknown        |

5. [model_manager](../model_managers/): manage training and generation of the model (e.g., how to use the first chunk as the seed chunk)

|           Parameter           |                                                                            Description                                                                            |
|:-----------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         `pretrain_dir`        |                    Set to the checkpoint folder if based on a pretrained public model.<br/>NULL otherwise. (The codebase will take care of that)                   |
|      `skip_chunk0_train`      |                 False if the first chunk is training from scratch (in most cases).<br/>True if the first chunk has pretrained models to load from.                 |
|       `pretrain_non_dp`       |         Cases without differential privacy.<br/>True if pretrain is required (using the first chunk).<br/>False if all chunks will be trained independently.        |
| `pretrain_non_dp_reduce_time` |                                     Cases without differential privacy.<br/>Time to save for other chunks except the first one.                                    |
|         `pretrain_dp`         | Cases with differential privacy.<br/>True if pretrain from a public pretrained model. `pretrain_dir` must be set in this case.<br/>False if training with naive DP. |
|             `run`             |                                                           Index to differentiate between different runs.                                                          |

6. [model](../models/): SOTA time-series GAN model [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)

    Please refer to [README of DoppelGANger](https://github.com/fjxmlzn/DoppelGANger#customize-doppelganger) for a detailed explanation of the parameters in this part.

7. generate: How to generate the data

|  Parameter  |                                 Description                                 |
|:-----------:|:---------------------------------------------------------------------------:|
|   `word2vec_vecSize` |     The size of embedded "word" for word2vec model (encoding port/protocol) |


8. default: [default configuration files](./default/) you may choose for preset datasets. Currently we support ["pcap.json"](./default/pcap.json) or ["netflow.json"](./default/netflow.json).
