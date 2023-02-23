We support multiple common data schemas and here are a few examples with corresponding configuration files. You may find the "nearest match" to start with.

**Note: across all examples, `iteration` are set to a small number to ensure a quick E2E test. For generating high-quality synthetic data, we recommend increasing `iteration` by your experience and computational resources.**

# Prerequiste
We support four different fields:
1. Bit field (encoded as bit strings) e.g., 
    ```JSON
    {
        "column": "srcip",
        "type": "integer",
        "encoding": "bit",
        "n_bits": 32
    }
    ```
   An optional property to this field is `truncate`, which is a boolean value with default `False`. If `truncate` is set to `true`, then we will truncate large integers and consider only the most significant `n_bits` bits. 

2. Word2Vec field (encoded as Word2Vec vectors), e.g.,
    ```JSON
    {
        "column": "srcport",
        "type": "integer",
        "encoding": "word2vec_port"
    }
    ```
3. Categorical field (encoded as one-hot encoding), e.g., 
    ```JSON
    {
        "column": "type",
        "type": "string",
        "encoding": "categorical"
    }
    ```
4. Continuous field, e.g.,
    ```JSON
    {
        "column": "pkt",
        "type": "float",
        "normalization": "ZERO_ONE",
        "log1p_norm": true
    }
    ```

# Dataset type 1: single-event
Single-event schema contains one timeseries per row.

## Data schema
| Timestamp (optional) | Metadata 1 | Metadata 2 | ... | Timeseries 1 | Timeseries 2 | ... |
|:--------------------:|:----------:|:----------:|:---:|:-------------:|:-------------:|:---:|
|          t1          |            |            |     |               |               |     |
|          t2          |            |            |     |               |               |     |
|          ...         |            |            |     |               |               |     |

## Examples
1. PCAP
    | Timestamp | Srcip | Dstip | Srcport | Dstport | Proto | Pkt_size | ... |
    |:---------:|:-----:|:-----:|:-------:|:-------:|:-----:|:--------:|:---:|
    |     t1    |       |       |         |         |       |          |     |
    |     t2    |       |       |         |         |       |          |     |
    |    ...    |       |       |         |         |       |          |     |

2. NetFlow ([configuration_file](netflow/config_example_netflow_nodp.json))

<!-- 3. [HAR dataset](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset) ([configuration_file]())
    |   | attitude.roll | attitude.pitch | attitude.yaw | userAcceleration.x | userAcceleration.y | userAcceleration.z | act | id  | weight | height | age  | gender | trial |
    |---|---------------|----------------|--------------|--------------------|--------------------|--------------------|-----|-----|--------|--------|------|--------|-------|
    | 0 | 1.528132      | -0.733896      | 0.696372     | 0.294894           | -0.184493          | 0.377542           | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |
    | 1 | 1.527992      | -0.716987      | 0.677762     | 0.219405           | 0.035846           | 0.114866           | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |
    | 2 | 1.527765      | -0.706999      | 0.670951     | 0.010714           | 0.134701           | -0.167808          | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   | -->



# [Dataset type 2: multi-event](./dg_table_row_per_sample/README.md)
Multi-event data schema contains multiple timeseries per row.

## Data Schema
| Metadata 1 | Metadata 2 | ... | {Timestamp (optional), Timeseries 1, Timeseries 2, ...} | {Timestamp (optional), Timeseries 1, Timeseries 2, ...} | ... |
|:----------:|:----------:|:---:|:-------------------------------------------------------:|:-------------------------------------------------------:|:---:|
|            |            |     |                                                         |                                                         |     |
|            |            |     |                                                         |                                                         |     |

## Examples
1. Wikipedia dataset ([configuration_file](./dg_table_row_per_sample/config_example_wiki.json))
    | Domain | Access type | Agent | {Date 1, page view} | {Date 2, page view} | ... |
    |:------:|:-----------:|:-----:|:-------------------:|:-------------------:|:---:|
    |        |             |       |                     |                     |     |
    |        |             |       |                     |                     |     |

