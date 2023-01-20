import csv
import os
import random
import tempfile
from typing import Dict, List

import numpy as np

from netshare.configs import get_config
from netshare.generate import generate_api
from netshare.input_adapters.input_adapter_api import get_canonical_data_dir
from netshare.learn import learn_api
from netshare.learn.utils.dataframe_utils import load_dataframe_chunks
from netshare.utils.field import ContinuousField, Field
from netshare.utils.logger import logger


def _get_fields_names(fields_list: List[Field]) -> List[str]:
    """
    This function returns the names of the given fields.
    """
    field_names = []
    for field in fields_list:
        if isinstance(field.name, list):
            field_names.extend(field.name)
        else:
            field_names.append(field.name)
    return field_names


def _denormalize_by_fields_list(
    normalized_data: np.ndarray, fields_list: List[Field], is_session_key: bool
) -> List[np.ndarray]:
    """
    This function executes field.denormalize for each of the given field.
    """
    denormalized_data = []
    dim = 0

    canonical_data_dir = get_canonical_data_dir()
    df, _ = load_dataframe_chunks(canonical_data_dir)

    for field in fields_list:
        if is_session_key:
            sub_data = normalized_data[:, dim : dim + field.getOutputDim()]
        else:
            sub_data = normalized_data[:, :, dim : dim + field.getOutputDim()]

        sub_data = field.denormalize(sub_data)
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
    filename: str,
) -> None:
    """
    This function dumps the given data to the given directory as a csv format.
    """
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, f"data_{filename}_{random.random()}.csv")
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        session_titles = _get_fields_names(session_key_fields)
        timeseries_titles = _get_fields_names(timeseries_fields)
        writer.writerow(session_titles + timeseries_titles)

        for i in range(session_key[0].shape[0]):
            session_data = [d[i] for d in session_key]
            # this if is here in parallel to the if in `reduce_samples`. It supports old flows.
            if len(timeseries) == 1:
                timeseries_data = timeseries[0][i].tolist()
            else:
                timeseries_data = [d[i][0] for d in timeseries]
            writer.writerow(session_data + timeseries_data)


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
        filename,
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
            filename=filename,
        )

    return output_folder
