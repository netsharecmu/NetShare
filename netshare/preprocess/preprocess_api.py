import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from netshare.configs import get_config
from netshare.utils.field import BitField, Field, FieldKey
from netshare.utils.logger import logger
from netshare.utils.paths import get_preprocessed_data_folder


def get_chunk_dir(chunk_id: Optional[int]) -> str:
    if get_config("global_config.n_chunks", default_value=1) == 1:
        if chunk_id == 0 or chunk_id is None:
            return get_preprocessed_data_folder()
        else:
            raise ValueError(
                f"Internal error: shouldn't have chunk_id={chunk_id} in this configuration"
            )
    if chunk_id is None:
        logger.warning("chunk_id is None, moved to a fallback chunk_id=0")
        chunk_id = 0
    chunk_dir = os.path.join(get_preprocessed_data_folder(), f"chunkid-{chunk_id}")
    os.makedirs(chunk_dir, exist_ok=True)
    return chunk_dir


def get_all_chunks_dirs() -> List[str]:
    """
    This function returns the directories of all the chunks.
    """
    if get_config("global_config.n_chunks", default_value=1) == 1:
        return [get_preprocessed_data_folder()]
    return [
        os.path.join(get_preprocessed_data_folder(), f"chunkid-{chunk_id}")
        for chunk_id in range(get_config("global_config.n_chunks"))
    ]


def get_word2vec_model_directory() -> str:
    return get_preprocessed_data_folder()


def create_dirs(chunk_id: int) -> None:
    """
    Create the directories for the preprocessed data.
    """
    os.makedirs(get_chunk_dir(chunk_id), exist_ok=True)


def write_raw_chunk(chunk: pd.DataFrame, chunk_id: int) -> None:
    """
    Writing the raw data of this chunk to its own directory.
    """
    chunk.to_csv(os.path.join(get_chunk_dir(chunk_id), "raw.csv"), index=False)


def write_data_train_npz(
    data_attribute: np.ndarray,
    data_feature: np.ndarray,
    global_max_flow_len: int,
    chunk_id: int,
) -> None:
    """
    TODO: Can someone help me document this function? What are the types of the data inside each file?
    """
    data_out_dir = get_chunk_dir(chunk_id)
    num_rows = data_attribute.shape[0]
    os.makedirs(os.path.join(data_out_dir, "data_train_npz"), exist_ok=True)
    gt_lengths = []
    for row_id in range(num_rows):
        gt_lengths.append(len(data_feature[row_id]))
        np.savez(
            os.path.join(data_out_dir, "data_train_npz", f"data_train_{row_id}.npz"),
            data_feature=data_feature[row_id],
            data_attribute=data_attribute[row_id],
            data_gen_flag=np.ones(len(data_feature[row_id])),
            global_max_flow_len=[
                global_max_flow_len
                if global_max_flow_len != 1
                else len(data_feature[row_id])
            ],
        )
    np.save(os.path.join(data_out_dir, "gt_lengths"), gt_lengths)


def _write_fields(
    field_type: str, fields: Dict[FieldKey, Field], chunk_id: int
) -> None:
    data_out_dir = get_chunk_dir(chunk_id)
    with open(os.path.join(data_out_dir, f"data_{field_type}_output.pkl"), "wb") as f:
        data_attribute_output = []
        for v in fields.values():
            if isinstance(v, BitField):
                data_attribute_output += v.getOutputType()
            else:
                data_attribute_output.append(v.getOutputType())
        pickle.dump(data_attribute_output, f)
    with open(os.path.join(data_out_dir, f"data_{field_type}_fields.pkl"), "wb") as f:
        pickle.dump(fields, f)


def write_attributes(metadata_fields: Dict[FieldKey, Field], chunk_id: int) -> None:
    """
    This function stores the attributes fields in two pickle files: one for the output types and one for the fields.
    # TODO: Attributes are cross-chunks, it doesn't make sense to have a chunk_id here.
    """
    _write_fields(field_type="attribute", fields=metadata_fields, chunk_id=chunk_id)


def write_features(timeseries_fields: Dict[FieldKey, Field], chunk_id: int) -> None:
    """
    Similar to write_attributes, but for the features.
    # TODO: Features are cross-chunks, it doesn't make sense to have a chunk_id here.
    """
    _write_fields(field_type="feature", fields=timeseries_fields, chunk_id=chunk_id)


def get_fields(field_type: str, chunk_id: Optional[int]) -> Dict[FieldKey, Field]:
    """
    This function loads the attributes fields from the pickle file.
    """
    with open(
        os.path.join(get_chunk_dir(chunk_id), f"data_{field_type}_fields.pkl"), "rb"
    ) as f:
        return pickle.load(f)  # type: ignore


def get_attributes_fields(chunk_id: Optional[int] = None) -> Dict[FieldKey, Field]:
    """
    This function returns the attributes fields that were stored in the preprocess phase.
    # TODO: Attributes are cross-chunks, it doesn't make sense to have a chunk_id here.
    """
    return get_fields("attribute", chunk_id)


def get_feature_fields(chunk_id: Optional[int] = None) -> Dict[FieldKey, Field]:
    """
    Similar to get_attributes_fields, but for the features.
    # TODO: Features are cross-chunks, it doesn't make sense to have a chunk_id here.
    """
    return get_fields("feature", chunk_id)
