import pandas as pd
import numpy as np
import pickle
import os
import csv
from tqdm import tqdm

from .pre_post_processor import PrePostProcessor
from netshare.utils.field import ContinuousField, DiscreteField
from netshare.utils.output import Normalization

EPS = 1e-8


class DGRowPerSamplePrePostProcessor(PrePostProcessor):
    def _pre_process(self, input_folder, output_folder, log_folder):
        # input is a file path
        file_path = input_folder

        original_df = pd.read_csv(file_path)

        # Remove missing rows.
        original_df.dropna(inplace=True)

        # Parse data.
        metadata_numpys = []
        metadata_fields = []
        for i, field in enumerate(self._config.metadata):
            if not isinstance(field.column, str):
                raise ValueError('"column" should be a string')
            this_df = original_df[field.column].astype(str)
            if 'regex' in field:
                this_df = this_df.str.extract(field.regex, expand=False)
            if field.type == 'string':
                choices = list(pd.unique(this_df))
                field_instance = DiscreteField(
                    choices=choices,
                    name=getattr(field, 'name', field.column))
                this_numpy = field_instance.normalize(this_df.to_numpy())
            elif field.type == 'float':
                this_df = this_df.astype(np.float64)
                this_numpy = this_df.to_numpy()
                this_numpy = this_numpy.reshape((this_df.shape[0], 1))
                field_instance = ContinuousField(
                    norm_option=getattr(Normalization, field.normalization),
                    min_x=this_numpy.min() - EPS,
                    max_x=this_numpy.max() + EPS,
                    dim_x=1,
                    name=getattr(field, 'name', field.column))
                this_numpy = field_instance.normalize(this_numpy)
            else:
                raise ValueError(f'Unknown field type {field.type}')
            metadata_numpys.append(this_numpy)
            metadata_fields.append(field_instance)
        metadata_numpy = np.concatenate(
            metadata_numpys, axis=1).astype(np.float64)
        print(f'List of metadata: '
              f'{list((k.dtype, k.shape) for k in metadata_numpys)}')
        print(f'Metadata type: {metadata_numpy.dtype}, '
              f'shape: {metadata_numpy.shape}')

        timeseries_numpys = []
        timeseries_fields = []
        for i, field in enumerate(self._config.timeseries):
            if not isinstance(field.columns, list):
                raise ValueError('"columns" should be a list')
            this_df = original_df[field.columns].astype(str)
            if 'regex' in field:
                for column in field.columns:
                    this_df[column] = this_df[column].str.extract(
                        field.regex, expand=False)
            if field.type == 'string':
                choices = list(pd.unique(this_df.values.ravel('K')))
                field_instance = DiscreteField(
                    choices=choices,
                    name=getattr(field, 'name', field.columns))
                this_numpy = field_instance.normalize(this_df.to_numpy())
                this_numpy = this_numpy.reshape(
                    (this_df.shape[0], len(field.columns), len(choices)))
            elif field.type == 'float':
                this_df = this_df.astype(np.float64)
                this_numpy = this_df.to_numpy()
                this_numpy = this_numpy.reshape(
                    (this_df.shape[0], len(field.columns), 1))
                if getattr(field, 'log1p_norm', False):
                    this_numpy = np.log1p(this_numpy)
                field_instance = ContinuousField(
                    norm_option=getattr(Normalization, field.normalization),
                    min_x=this_numpy.min() - EPS,
                    max_x=this_numpy.max() + EPS,
                    dim_x=1,
                    name=getattr(field, 'name', field.columns))
                this_numpy = field_instance.normalize(this_numpy)
            else:
                raise ValueError(f'Unknown field type {field.type}')
            timeseries_numpys.append(this_numpy)
            timeseries_fields.append(field_instance)
        timeseries_numpy = np.concatenate(timeseries_numpys, axis=2).astype(
            np.float64)
        print(f'List of timeseries: '
              f'{list((k.dtype, k.shape) for k in timeseries_numpys)}')
        print(f'Timeseries type: {timeseries_numpy.dtype}, '
              f'shape: {timeseries_numpy.shape}')

        # Randomly select the required number of samples.
        np.random.seed(getattr(self._config, 'random_seed', 0))
        ids = np.random.permutation(metadata_numpy.shape[0])
        metadata_train_numpy = metadata_numpy[
            ids[:self._config.num_train_samples]]
        timeseries_train_numpy = timeseries_numpy[
            ids[:self._config.num_train_samples]]

        print(f'Metadata train type: {metadata_train_numpy.dtype}, '
              f'shape: {metadata_train_numpy.shape}')
        print(f'Timeseries train type: {timeseries_train_numpy.dtype}, '
              f'shape: {timeseries_train_numpy.shape}')

        # Write files
        with open(os.path.join(
                output_folder, 'data_attribute_output.pkl'), 'wb') as f:
            pickle.dump([v.getOutputType() for v in metadata_fields], f)
        with open(os.path.join(
                output_folder, 'data_feature_output.pkl'), 'wb') as f:
            pickle.dump([v.getOutputType() for v in timeseries_fields], f)
        with open(os.path.join(
                output_folder, 'data_attribute_fields.pkl'), 'wb') as f:
            pickle.dump(metadata_fields, f)
        with open(os.path.join(
                output_folder, 'data_feature_fields.pkl'), 'wb') as f:
            pickle.dump(timeseries_fields, f)
        npz_folder = os.path.join(output_folder, 'data_train_npz')
        os.makedirs(npz_folder)
        for i in range(metadata_train_numpy.shape[0]):
            np.savez(
                os.path.join(npz_folder, f'data_train_{i}.npz'),
                data_feature=timeseries_train_numpy[i],
                data_attribute=metadata_train_numpy[i],
                data_gen_flag=np.ones(timeseries_train_numpy.shape[1]),
                global_max_flow_len=[timeseries_train_numpy.shape[1]])

        return True

    def _post_process(self, input_folder, output_folder,
                      pre_processed_data_folder, log_folder):
        with open(os.path.join(
                pre_processed_data_folder,
                'data_attribute_fields.pkl'), 'rb') as f:
            metadata_fields = pickle.load(f)
        with open(os.path.join(
                pre_processed_data_folder,
                'data_feature_fields.pkl'), 'rb') as f:
            timeseries_fields = pickle.load(f)
        sub_folders = os.listdir(input_folder)
        for sub_folder in sub_folders:
            data_path = os.path.join(input_folder, sub_folder, 'data.npz')
            data = np.load(data_path)
            unnormalized_timeseries = data['data_feature']
            unnormalized_metadata = data['data_attribute']
            data_gen_flag = data['data_gen_flag']
            timeseries = []
            metadata = []
            dim = 0
            for field_i, field in enumerate(metadata_fields):
                sub_metadata = field.denormalize(
                    unnormalized_metadata[
                        :, dim: dim + field.getOutputType().dim])
                if getattr(self._config.metadata[field_i], 'log1p_norm',
                           False):
                    sub_metadata = np.exp(sub_metadata) - 1
                if isinstance(field, ContinuousField):
                    sub_metadata = sub_metadata[:, 0]
                metadata.append(sub_metadata)
                dim += field.getOutputType().dim
            assert dim == unnormalized_metadata.shape[1]

            timeseries = []
            dim = 0
            for field_i, field in enumerate(timeseries_fields):
                sub_timeseries = field.denormalize(
                    unnormalized_timeseries[
                        :, :, dim: dim + field.getOutputType().dim])
                if getattr(self._config.timeseries[field_i], 'log1p_norm',
                           False):
                    sub_timeseries = np.exp(sub_timeseries) - 1
                if isinstance(field, ContinuousField):
                    sub_timeseries = sub_timeseries[:, :, 0]
                timeseries.append(sub_timeseries)
                dim += field.getOutputType().dim
            assert dim == unnormalized_timeseries.shape[2]

            csv_folder = os.path.join(output_folder, sub_folder)
            os.makedirs(csv_folder)
            csv_path = os.path.join(csv_folder, 'data.csv')
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [field.name for field in metadata_fields] +
                    [column_name for field in timeseries_fields
                     for column_name in field.name])
                for i in tqdm(range(unnormalized_timeseries.shape[0])):
                    writer.writerow(
                        [d[i] for d in metadata] +
                        [sd
                         for d in timeseries
                         for sd in d[i][:int(np.sum(data_gen_flag[i]))]])
        return True
