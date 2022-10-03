Here is a quick guide to add new datasets:

1. Implement a new Prepostprocessor in NetShare/netshare/pre_post_processors
2. Create a new file called xxx_pre_post_processor.py and define a new class XXXPrePostProcessor.
3. Import it in NetShare/netshare/pre_post_processors/__init__.py and put the classname into __all__
4. Preprocess: Normalize continuous fields, one-hot / embedding categorical fields...
5. Postprocess: Add logic to map synthetic data back to the origin domain.

You may refer to zeek connection log preprocess as an example:
- dataset: traces/zeek_conn_log/raw.csv
- script: netshare/pre_post_processors/netshare/zeeklog_pre_post_processor.py
