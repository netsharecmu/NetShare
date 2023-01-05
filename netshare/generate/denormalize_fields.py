import csv
import os
import random
import tempfile
from typing import List

import numpy as np

from netshare.generate import generate_api
from netshare.learn import learn_api
from netshare.utils.field import ContinuousField, Field


def _denormalize_by_fields_list(
    normalized_data: np.ndarray, fields_list: List[Field], is_session_key: bool
) -> List[np.ndarray]:
    """
    This function executes field.denormalize for each of the given field.
    """
    denormalized_data = []
    dim = 0
    for field in fields_list:
        if is_session_key:
            sub_data = normalized_data[:, dim : dim + field.getOutputDim()]
        else:
            sub_data = normalized_data[:, :, dim : dim + field.getOutputDim()]
        sub_data = field.denormalize(sub_data)
        if isinstance(field, ContinuousField):
            sub_data = sub_data[:, 0] if is_session_key else sub_data[:, :, 0]
        denormalized_data.append(sub_data)
        dim += field.getOutputDim()
    return denormalized_data


def write_to_csv(
    csv_folder: str,
    session_key_fields: List[Field],
    timeseries_fields: List[Field],
    session_key: List[np.ndarray],
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
            [field.name for field in session_key_fields]
            + [column_name for field in timeseries_fields for column_name in field.name]
        )
        for i in range(session_key[0].shape[0]):
            writer.writerow(
                [d[i] for d in session_key]
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

    session_key_fields = list(learn_api.get_attributes_fields().values())
    timeseries_fields = list(learn_api.get_feature_fields().values())

    for (
        unnormalized_timeseries,
        unnormalized_session_key,
        data_gen_flag,
        sub_folder,
    ) in generate_api.get_raw_generated_data():
        session_key = _denormalize_by_fields_list(
            unnormalized_session_key, session_key_fields, is_session_key=True
        )
        timeseries = _denormalize_by_fields_list(
            unnormalized_timeseries, timeseries_fields, is_session_key=False
        )
        write_to_csv(
            csv_folder=os.path.join(output_folder, sub_folder),
            session_key_fields=session_key_fields,
            timeseries_fields=timeseries_fields,
            session_key=session_key,
            timeseries=timeseries,
            data_gen_flag=data_gen_flag,
        )

    return output_folder
