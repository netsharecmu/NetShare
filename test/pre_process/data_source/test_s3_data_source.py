from netshare.pre_process.data_source.s3_data_source import S3DataSource
from moto import mock_s3
import boto3


@mock_s3
def test_fetch_data(tmp_path):
    data_source = S3DataSource()

    # Create mock bucket
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    client.put_object(Bucket="test-bucket", Key="file1.txt", Body="file1")
    client.put_object(Bucket="test-bucket", Key="subdir/file2.txt", Body="file2")

    target = tmp_path / "target"
    target.mkdir()

    data_source.fetch_data({"bucket_name": "test-bucket"}, str(target))

    # Check that the files were copied
    assert (target / "file1.txt").read_text() == "file1"
    assert (target / "subdir_file2.txt").read_text() == "file2"
