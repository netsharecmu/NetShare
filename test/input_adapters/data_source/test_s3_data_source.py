import os

import boto3
from moto import mock_s3

from netshare.configs import set_config
from netshare.input_adapters.data_source.s3_data_source import S3DataSource


@mock_s3
def test_fetch_data(tmp_path):
    data_source = S3DataSource()

    # Create mock bucket
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    client.put_object(Bucket="test-bucket", Key="file1.txt", Body="file1")
    client.put_object(Bucket="test-bucket", Key="subdir/file2.txt", Body="file2")

    set_config({"bucket_name": "test-bucket"})
    target = data_source.fetch_data()

    # Check that the files were copied
    assert open(os.path.join(target, "file1.txt"), "r").read() == "file1"
    assert open(os.path.join(target, "subdir_file2.txt"), "r").read() == "file2"
