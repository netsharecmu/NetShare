import csv
import os
import random
import tempfile
from typing import List

import numpy as np

from netshare.models import model_api
from netshare.preprocess import preprocess_api
from netshare.utils.field import ContinuousField, Field


def _denormalize_by_fields_list(
    normalized_data: np.ndarray, fields_list: List[Field], is_metadata: bool
) -> List[np.ndarray]:
    """
    This function executes field.denormalize for each of the given field.
    """
    denormalized_data = []
    dim = 0
    for field in fields_list:
        if is_metadata:
            sub_data = normalized_data[:, dim : dim + field.getOutputDim()]
        else:
            sub_data = normalized_data[:, :, dim : dim + field.getOutputDim()]
        sub_data = field.denormalize(sub_data)
        if isinstance(field, ContinuousField):
            sub_data = sub_data[:, 0] if is_metadata else sub_data[:, :, 0]
        denormalized_data.append(sub_data)
        dim += field.getOutputDim()
    return denormalized_data


def write_to_csv(
    csv_folder: str,
    metadata_fields: List[Field],
    timeseries_fields: List[Field],
    metadata: List[np.ndarray],
    timeseries: List[np.ndarray],
    data_gen_flag: np.ndarray,
) -> None:
    """
    This function dumps the given data to the given directory as a csv format.
    """
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, f"data_{random.random()}.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [field.name for field in metadata_fields]
            + [column_name for field in timeseries_fields for column_name in field.name]
        )
        for i in range(metadata[0].shape[0]):
            writer.writerow(
                [d[i] for d in metadata]
                + [
                    sd
                    for d in timeseries
                    for sd in d[i][: int(np.sum(data_gen_flag[i]))]
                ]
            )


def denormalize_fields() -> str:
    """
    This function denormalizes the data in the generated_data folder using the attributes
        and features fields that were created in the pre-process step.
    Last, it writes the denormalized data to a csv file under the same directory hierarchy as the created data.

    :return: the path to the denormalized data.
    """
    output_folder = tempfile.mkdtemp()

    metadata_fields = list(preprocess_api.get_attributes_fields().values())
    timeseries_fields = list(preprocess_api.get_feature_fields().values())

    for (
        unnormalized_timeseries,
        unnormalized_metadata,
        data_gen_flag,
        sub_folder,
    ) in model_api.get_generated_data():
        metadata = _denormalize_by_fields_list(
            unnormalized_metadata, metadata_fields, is_metadata=True
        )
        timeseries = _denormalize_by_fields_list(
            unnormalized_timeseries, timeseries_fields, is_metadata=False
        )
        write_to_csv(
            csv_folder=os.path.join(output_folder, sub_folder),
            metadata_fields=metadata_fields,
            timeseries_fields=timeseries_fields,
            metadata=metadata,
            timeseries=timeseries,
            data_gen_flag=data_gen_flag,
        )

    return output_folder
