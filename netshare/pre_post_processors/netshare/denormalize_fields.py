import csv
import os
import json
import random
import pickle
from typing import Dict, List

import numpy as np
from config_io import Config
from tqdm import tqdm

from netshare.utils.logger import logger


def _get_fields_names(fields_list):
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
    normalized_data,
    fields_list,
    is_session_key
):
    """
    This function executes field.denormalize for each of the given field.
    """
    denormalized_data = []
    dim = 0

    for field in fields_list:
        if is_session_key:
            sub_data = normalized_data[:, dim: dim + field.dim_x]
        else:
            sub_data = normalized_data[:, :, dim: dim + field.dim_x]

        sub_data = field.denormalize(sub_data)

        # For session key, if shape looks like (n, ), change it to (n, 1) for consistency
        if is_session_key and len(sub_data.shape) == 1:
            sub_data = np.expand_dims(sub_data, axis=1)
        # For timeseries, if shape looks like (i, j), change it to (i, j, 1) for consistency
        if not is_session_key and len(sub_data.shape) == 2:
            sub_data = np.expand_dims(sub_data, axis=2)
        denormalized_data.append(sub_data)
        dim += field.dim_x
    return denormalized_data


def write_to_csv(
    csv_folder,
    session_key_fields,
    timeseries_fields,
    session_key,
    timeseries,
    data_gen_flag,
    filename,
    config,
) -> None:
    """
    This function dumps the given data to the given directory as a csv format.
    `data_gen_flag` is an indicator showing if the time series for this session
    has ended in this time step.
    """
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, filename)
    # change session key shape to #session * #attributes
    session_key_numpy = np.array(np.concatenate(session_key, axis=1))
    # change timeseries shape to #session * #time_steps * #features
    timeseries_numpy = np.array(np.concatenate(timeseries, axis=2))

    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        raw_metadata_field_names = [
            col.column for col in (config["metadata"])
        ]
        raw_timeseries_filed_names = [
            col.column for col in config["timeseries"]]
        session_titles = [
            f for i, f in enumerate(_get_fields_names(session_key_fields))
            if f in raw_metadata_field_names]
        session_titles_idx = [
            i for i, f in enumerate(_get_fields_names(session_key_fields))
            if f in raw_metadata_field_names]
        timeseries_titles = [
            f
            for i, f in enumerate(_get_fields_names(timeseries_fields))
            if f in raw_timeseries_filed_names
        ]
        timeseries_titles_idx = [
            i
            for i, f in enumerate(_get_fields_names(timeseries_fields))
            if f in raw_timeseries_filed_names
        ]

        if config["timestamp"].get("generation", False):
            timeseries_titles.append(config["timestamp"]["column"])
            if config["timestamp"]["encoding"] == "interarrival":
                # Find `flow_start` and `interarrival_within_flow` index
                flow_start_idx, interarrival_within_flow_idx = None, None
                for idx, field_name in enumerate(
                        _get_fields_names(session_key_fields)):
                    if field_name == "flow_start":
                        flow_start_idx = idx
                        break
                for idx, field_name in enumerate(
                        _get_fields_names(timeseries_fields)):
                    if field_name == "interarrival_within_flow":
                        interarrival_within_flow_idx = idx
                        break
                if flow_start_idx is None or interarrival_within_flow_idx is None:
                    raise ValueError(
                        "Using `interarrival` encoding: `flow_start` or `interarrival_field` not found!"
                    )

                # convert interarrival to raw timestamp
                interarrival_cumsum = np.cumsum(
                    timeseries_numpy[:, :, interarrival_within_flow_idx].astype(
                        float),
                    axis=1)
                # first packet has 0.0 interarrival
                interarrival_cumsum[:, 0] = 0.0
                flow_start_expand = (
                    np.array(
                        [
                            session_key_numpy[:, flow_start_idx],
                        ]
                        * interarrival_cumsum.shape[1]
                    )
                    .transpose()
                    .astype(float)
                )
                timestamp_matrix = np.expand_dims(
                    np.add(flow_start_expand, interarrival_cumsum), axis=2
                )
                timeseries_numpy = np.concatenate(
                    (timeseries_numpy, timestamp_matrix), axis=2
                )
                timeseries_titles_idx.append(timeseries_numpy.shape[2] - 1)

        writer.writerow(session_titles + timeseries_titles)

        session_key_set = set()
        for (
            data_gen_per_session,
            session_data_per_session,
            timeseries_per_session,
        ) in zip(
            data_gen_flag,
            # remove cols not in raw data
            session_key_numpy[:, session_titles_idx],
            timeseries_numpy[
                :, :, timeseries_titles_idx
            ],  # remove cols not in raw data
        ):
            session_data_per_session = session_data_per_session.tolist()
            # remove duplicated session keys
            if tuple(session_data_per_session) in session_key_set:
                logger.debug(
                    "Session key {session_data_per_session} already exists!")
                continue
            session_key_set.add(tuple(session_data_per_session))
            for j in range(data_gen_per_session.shape[0]):
                if data_gen_per_session[j] == 1.0:
                    timeseries_data = timeseries_per_session[j].tolist()
                    writer.writerow(session_data_per_session + timeseries_data)


def denormalize_fields(
    config_pre_post_processor,
    pre_processed_data_folder,
    generated_data_folder,
    post_processed_data_folder
):
    """
    This function denormalizes the data in the generated_data folder using the attributes and features fields that were created in the pre-process step.
    Last, it writes the denormalized data to a csv file under the same directory hierarchy as the created data.

    :return: the path to the denormalized data.
    """
    with open(os.path.join(generated_data_folder, "configs_generate.json"), 'r') as f:
        data = json.load(f)
        configs = data["configs"]
        config_group_list = data["config_group_list"]

    for config in tqdm(configs):
        with open(os.path.join(
            pre_processed_data_folder,
            f"chunkid-{config['chunk_id']}",
            "data_attribute_fields.pkl"
        ), 'rb') as f:
            session_key_fields = list(pickle.load(f))

        with open(os.path.join(
            pre_processed_data_folder,
            f"chunkid-{config['chunk_id']}",
            "data_feature_fields.pkl"
        ), 'rb') as f:
            timeseries_fields = list(pickle.load(f))

        # Each configuration has multiple iteration ckpts
        per_chunk_basedir = os.path.join(
            config["eval_root_folder"],
            "feat_raw", f"chunk_id-{config['chunk_id']}")
        for f in os.listdir(per_chunk_basedir):
            if not f.endswith(".npz"):
                continue
            data = np.load(os.path.join(per_chunk_basedir, f))
            unnormalized_session_key = data["data_attribute"]
            unnormalized_timeseries = data["data_feature"]
            data_gen_flag = data["data_gen_flag"]

            session_key = _denormalize_by_fields_list(
                unnormalized_session_key, session_key_fields,
                is_session_key=True)
            timeseries = _denormalize_by_fields_list(
                unnormalized_timeseries, timeseries_fields, is_session_key=False
            )

            csv_root_folder = os.path.join(
                config["eval_root_folder"], "syn_dfs")
            csv_filename = f.replace(".npz", ".csv")
            write_to_csv(
                csv_folder=os.path.join(
                    csv_root_folder, f"chunk_id-{config['chunk_id']}"
                ),
                session_key_fields=session_key_fields,
                timeseries_fields=timeseries_fields,
                session_key=session_key,
                timeseries=timeseries,
                data_gen_flag=data_gen_flag,
                filename=csv_filename,
                config=config_pre_post_processor,
            )
