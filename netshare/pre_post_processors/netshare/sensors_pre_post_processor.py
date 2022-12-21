from typing import Dict, List
import shutil
import os

import pandas as pd
import numpy as np

from netshare.pre_post_processors import PrePostProcessor


TIME_COLUMN = "sequence"
METADATA_COLUMN = "metadata"


class SensorsPrePostProcessor(PrePostProcessor):
    def _validate_input(self, input_folder: str) -> None:
        if self._config["dataset_type"] != "csv":
            raise ValueError("dataset_type not supported")
        if not input_folder.endswith(".csv") and not os.path.isdir(input_folder):
            raise ValueError("Given path is not a csv file and not a directory")
        if self._config["encode_IP"] not in ["bit", "word2vec"]:
            raise ValueError("IP can be only encoded as `bit` or `word2vec`!")

    def _load_csv(self, input_folder: str) -> List[pd.DataFrame]:
        if input_folder.endswith(".csv"):
            csv_chunks = [pd.read_csv(input_folder)]
        else:
            csv_chunks = [
                pd.read_csv(
                    os.path.join(input_folder, filename), index_col=None, header=0
                )
                for filename in os.listdir(input_folder)
            ]
        return [csv.rename(columns={"Unnamed: 0": TIME_COLUMN}) for csv in csv_chunks]

    def _get_metadata_to_chunk_ids_mapping(
        self, df_chunks: List[pd.DataFrame]
    ) -> Dict[str, List[int]]:
        return {str(index): [index] for index, _ in enumerate(df_chunks)}

    def _dump_results(
        self, csv_dataframes: List[pd.DataFrame], output_folder: str
    ) -> None:
        data_out_dir = os.path.join(output_folder, f"chunkid-0")
        os.makedirs(data_out_dir, exist_ok=True)
        os.makedirs(os.path.join(data_out_dir, "data_train_npz"), exist_ok=True)
        for index, df in enumerate(csv_dataframes):
            df[METADATA_COLUMN] = str(index)
        pd.concat(csv_dataframes, axis=0, ignore_index=True).to_csv(
            os.path.join(data_out_dir, "raw.csv"), index=False
        )

        os.makedirs(os.path.join(data_out_dir, "data_train_npz"), exist_ok=True)
        for row_id in range(len(csv_dataframes)):
            np.savez(
                os.path.join(
                    data_out_dir, "data_train_npz", f"data_train_{row_id}.npz"
                ),
                data_feature=[],
                data_attribute=[],
                data_gen_flag=[],
                global_max_flow_len=[],
            )

    def _pre_process(self, input_folder, output_folder, log_folder):
        self._validate_input(input_folder)

        csv_dataframes = self._load_csv(input_folder)

        self._dump_results(csv_dataframes, output_folder)

        return True

    def _post_process(
        self, input_folder, output_folder, pre_processed_data_folder, log_folder
    ):
        shutil.copyfile(
            os.path.join(input_folder, "best_syn_dfs", "syn.csv"),
            os.path.join(output_folder, "syn.csv"),
        )
        return True
