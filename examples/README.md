We support multiple common data schemas and here are a few examples with corresponding configuration files. You may find the "nearest match" to start with.

# Type 1: single-event
Single-event schema contains one timeseries per row.

## Data schema
| Timestamp (optional) | Metadata 1 | Metadata 2 | ... | Timeseries 1 | Timeseries 2 | ... |
|:--------------------:|:----------:|:----------:|:---:|:-------------:|:-------------:|:---:|
|          t1          |            |            |     |               |               |     |
|          t2          |            |            |     |               |               |     |
|          ...         |            |            |     |               |               |     |

## Examples
### PCAP
| Timestamp | Srcip | Dstip | Srcport | Dstport | Proto | Pkt_size | ... |
|:---------:|:-----:|:-----:|:-------:|:-------:|:-----:|:--------:|:---:|
|     t1    |       |       |         |         |       |          |     |
|     t2    |       |       |         |         |       |          |     |
|    ...    |       |       |         |         |       |          |     |

### Configuration file (tentative)
```Json
"timestamp": {
    "column": "ts",
    "generation": true,
    "encoding": "interarrival",
    "normalization": "ZERO_ONE"
},
"word2vec": {
    "vec_size": 10,
    "model_name": "word2vec_vecSize",
    "annoy_n_trees": 100,
    "pretrain_model_path": null
},
"metadata": [
    {
        "column": "srcip",
        "type": "integer",
        "encoding": "bit",
        "n_bits": 32
    },
    {
        "column": "dstip",
        "type": "integer",
        "encoding": "bit",
        "n_bits": 32
    },
    {
        "column": "srcport",
        "type": "integer",
        "encoding": "word2vec_port"
    },
    {
        "column": "dstport",
        "type": "integer",
        "encoding": "word2vec_port"
    },
    {
        "column": "proto",
        "type": "string",
        "encoding": "word2vec_proto"
    }
],
"timeseries": [
    {
        "column": "td",
        "type": "float",
        "normalization": "ZERO_ONE",
        "log1p_norm": true
    },
    {
        "column": "pkt",
        "type": "float",
        "normalization": "ZERO_ONE",
        "log1p_norm": true
    },
    {
        "column": "byt",
        "type": "float",
        "normalization": "ZERO_ONE",
        "log1p_norm": true
    },
    {
        "column": "type",
        "type": "string"
    }
]
```

### [HAR dataset](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset)
|   | attitude.roll | attitude.pitch | attitude.yaw | userAcceleration.x | userAcceleration.y | userAcceleration.z | act | id  | weight | height | age  | gender | trial |
|---|---------------|----------------|--------------|--------------------|--------------------|--------------------|-----|-----|--------|--------|------|--------|-------|
| 0 | 1.528132      | -0.733896      | 0.696372     | 0.294894           | -0.184493          | 0.377542           | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |
| 1 | 1.527992      | -0.716987      | 0.677762     | 0.219405           | 0.035846           | 0.114866           | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |
| 2 | 1.527765      | -0.706999      | 0.670951     | 0.010714           | 0.134701           | -0.167808          | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |
| 3 | 1.516768      | -0.704678      | 0.675735     | -0.008389          | 0.136788           | 0.094958           | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |
| 4 | 1.493941      | -0.703918      | 0.672994     | 0.199441           | 0.353996           | -0.044299          | 0.0 | 0.0 | 102.0  | 188.0  | 46.0 | 1.0    | 1.0   |

### Configuration file (tentative)
```Json
"timestamp": {
    "column": null,
    "generation": null,
    "encoding": null,
    "normalization": null
},
"metadata": [
    {
        "column": "weight",
        "type": "float",
        "normalization": "ZERO_ONE"
    },
    {
        "column": "height",
        "type": "float",
        "normalization": "ZERO_ONE"
    },
    {
        "column": "age",
        "type": "float",
        "normalization": "ZERO_ONE"
    },
    {
        "column": "gender",
        "type": "float",
        "normalization": "ZERO_ONE"
    },
    {
        "column": "trial",
        "type": "string"
    }
],
"timeseries": [
    {
        "column": "attitude.roll",
        "type": "float",
        "normalization": "ZERO_ONE",
    },
    {
        "column": "attitude.yaw",
        "type": "float",
        "normalization": "ZERO_ONE",
    },
    {
        "column": "attitude.pitch",
        "type": "float",
        "normalization": "ZERO_ONE",
    },
    {
        "column": "userAcceleration.x",
        "type": "float",
        "normalization": "ZERO_ONE",
    },
    {
        "column": "userAcceleration.y",
        "type": "float",
        "normalization": "ZERO_ONE",
    },
    {
        "column": "userAcceleration.z",
        "type": "float",
        "normalization": "ZERO_ONE",
    },
    {
        "column": "act",
        "type": "string"
    }
]
```

# Type 2: multi-event
Multi-event data schema contains multiple timeseries per row.

| Metadata 1 | Metadata 2 | ... | {Timestamp (optional), Timeseries 1, Timeseries 2, ...} | {Timestamp (optional), Timeseries 1, Timeseries 2, ...} | ... |
|:----------:|:----------:|:---:|:-------------------------------------------------------:|:-------------------------------------------------------:|:---:|
|            |            |     |                                                         |                                                         |     |
|            |            |     |                                                         |                                                         |     |

## Examples
### Wikipedia dataset
| Domain | Access type | Agent | {Date 1, page view} | {Date 2, page view} | ... |
|:------:|:-----------:|:-----:|:-------------------:|:-------------------:|:---:|
|        |             |       |                     |                     |     |
|        |             |       |                     |                     |     |
