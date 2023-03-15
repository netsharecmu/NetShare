# Datasets in Table Format

This folder contains examples and instructions for datasets where each row is one sample (containing metadata and timeseries) in table format (e.g., CSV, XLSX).

## A Quick Run

This example uses [Wikipedia Web Traffic Dataset](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data). The data is in CSV format, and each row contains daily page views of a website from Jul. 1st, 2015 to Dec. 31st, 2016, and their associated metadata (e.g., domain name, access type, agent). The goal of this example is to train a [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger) model (without the new features in [NetShare](https://github.com/netsharecmu/NetShare)) on this dataset, and generate a synthetic dataset. The data processing steps and the training steps follow the original experiments in [DoppelGANger paper](http://arxiv.org/abs/1909.13403).

The steps for running the example:

* Follow the steps in [https://github.com/netsharecmu/NetShare](https://github.com/netsharecmu/NetShare) to install NetShare.
* Download `train_1.csv` from [kaggle](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data) and put it at `traces/wiki/train_1.csv`.
* Run the example

```
cd examples/dg_table_row_per_sample
python driver.py
```
* Generated data will be at `<work_folder>/post_processed_data/iteration_id-199999/data.csv`, where `work_folder` is the path specified in `driver.py` (which you may want to change).

## Your Own Dataset
The code supports any dataset where each row is one sample (containing metadata and timeseries) in table format (e.g., CSV, XLSX). The only change needed is the [config](https://github.com/netsharecmu/NetShare/tree/master/examples/dg_table_row_per_sample/config_example_wiki.json) file. The key parameters are:

* `metadata` contains the list of table columns that will be treated as metadata of the sample (please see [DoppelGANger paper](http://arxiv.org/abs/1909.13403) or [DoppelGANger code](https://github.com/fjxmlzn/DoppelGANger) for the detailed explanation of metadata).
	* `column`: The column name of the table that will be treated as this metadata.
	* `regex` [optional]: If the value of the metadata is not exactly the value in the table cell, you can write a regular expression here to extract the value.
	* `type`: `string` or `float`, indicating the type of this metadata.
	* `name` [optional]: The name of this metadata. This will be the column name for this metadata in the generated table. If not provided, the value of `column` will be used.
* `timeseries` contains the list of table columns that will be treated as timeseries of the sample (please see [DoppelGANger paper](http://arxiv.org/abs/1909.13403) or [DoppelGANger code](https://github.com/fjxmlzn/DoppelGANger) for the detailed explanation of timeseries).
	* `columns`: The list of column names of the table that will be treated as timeseries part of the same.
	* `regex` [optional]: Same as the one in `metadata`.
	* `type`: Same as the one in `metadata`.
	* `name` [optional]: The list of names that be used as the column names for this timeseries in the generated table. If not provided, the value of `columns` will be used.

