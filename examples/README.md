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
"metadata": [
    {
        "column": "srcip",
        "type": "bit"
    },
    {
        "column": "dstip",
        "type": "bit"
    },
    {
        "column": "srcport",
        "type": "word2vec"
    },
    {
        "column": "dstport",
        "type": "word2vec"
    },
    {
        "column": "proto",
        "type": "word2vec"
    }
],
"timeseries": []
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
