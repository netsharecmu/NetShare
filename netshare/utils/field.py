import json
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from config_io import Config

from netshare.learn.utils.word2vec_embedding import (
    get_original_objs,
    get_vector,
    get_word2vec_type_col,
)
from netshare.utils.paths import (
    get_annoy_dict_idx_ele_for_word2vec,
    get_annoyIndex_for_word2vec,
)

from .output import Normalization, Output, OutputType

EPS = 1e-8
FieldKey = Tuple[str, str, str]


class Field(object):
    def __init__(self, name: Union[str, List[str]], log1p_norm: bool = False):
        self.name = name
        self.log1p_norm = log1p_norm

    def normalize(self, x):
        raise NotImplementedError

    def denormalize(self, x):
        raise NotImplementedError

    def get_output_type(self):
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


class ContinuousField(Field):
    def __init__(
        self,
        name: Union[str, List[str]],
        norm_option: Normalization,
        min_x: float,
        max_x: float,
        dim_x: int = 1,
        log1p_norm: bool = False,
    ) -> None:
        super(ContinuousField, self).__init__(name=name, log1p_norm=log1p_norm)

        self.min_x = min_x
        self.max_x = max_x
        self.norm_option = norm_option
        self.dim_x = dim_x
        if self.log1p_norm:
            # If set log1p_norm, we need to re-calculate the min_x and max_x after doing
            # log1p. Since the denormalizing step would first do min-max denomalization
            # and then expm1. Using stale min_x and max_x would cause the synthetic
            # value larger than expected and is possible to have np.inf
            self.min_x = np.log1p(self.min_x)
            self.max_x = np.log1p(self.max_x)

    # Normalize x in [a, b]: x' = (b-a)(x-min x)/(max x - minx) + a
    def normalize(self, x):
        if x.shape[-1] != self.dim_x:
            raise ValueError(
                f"Dimension is {x.shape[-1]}. Expected dimension is {self.dim_x}"
            )
        if self.log1p_norm:
            x = np.log1p(x)

        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            if self.max_x - self.min_x == 0:
                raise Exception("Not enough data to proceed!")
            return np.asarray((x - self.min_x) / (self.max_x - self.min_x))

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            if self.max_x - self.min_x == 0:
                raise Exception("Not enough data to proceed!")
            return np.asarray(2 * (x - self.min_x) / (self.max_x - self.min_x) - 1)
        else:
            raise Exception("Not valid normalization option!")

    def denormalize(self, norm_x):
        if self.max_x is None or self.min_x is None:
            return norm_x  # This is a word2vec field
        if norm_x.shape[-1] != self.get_output_dim():
            raise ValueError(
                f"Dimension is {norm_x.shape[-1]}. Expected dimension is {self.get_output_dim()}"
            )
        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            to_return = norm_x * float(self.max_x - self.min_x) + self.min_x

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            to_return = (norm_x + 1) / 2.0 * float(self.max_x - self.min_x) + self.min_x

        else:
            raise Exception("Not valid normalization option!")
        if self.log1p_norm:

            to_return = np.expm1(to_return)
        return to_return

    def get_output_type(self):
        return Output(
            type_=OutputType.CONTINUOUS,
            dim=self.get_output_dim(),
            normalization=self.norm_option,
        )

    def get_output_dim(self) -> int:
        return self.dim_x


class DiscreteField(Field):
    def __init__(self, choices, *args, **kwargs):
        super(DiscreteField, self).__init__(*args, **kwargs)

        if not isinstance(choices, list):
            raise Exception("choices should be a list")
        self.choices = choices

    def normalize(self, x):
        if not isinstance(x, (list, np.ndarray)):
            norm_x = [x]
        else:
            norm_x = x
        norm_x = pd.DataFrame(norm_x).astype(
            pd.CategoricalDtype(categories=self.choices)
        )
        norm_x = pd.get_dummies(norm_x).to_numpy()
        if not isinstance(x, (list, np.ndarray)):
            norm_x = norm_x[0]

        return norm_x

    def denormalize(self, norm_x):
        index = np.argmax(norm_x, axis=-1)

        return np.asarray(self.choices)[index]

    def get_output_type(self):
        return Output(type_=OutputType.DISCRETE, dim=self.get_output_dim())

    def get_output_dim(self) -> int:
        return len(self.choices)


class RegexField(DiscreteField):
    def __init__(self, regex, choices, *args, **kwargs):
        super(RegexField, self).__init__(choices, *args, **kwargs)
        self.regex = regex

    def normalize(self, x):
        x = pd.Series(x.reshape((x.shape[0],))).str.extract(self.regex, expand=False)
        if not self.choices:
            self.choices = list(pd.unique(x))
        return super(RegexField, self).normalize(x.to_numpy())


class BitField(Field):
    def __init__(self, num_bits, *args, truncate=False, **kwargs):
        super(BitField, self).__init__(*args, **kwargs)

        self.num_bits = num_bits
        self.truncate = truncate

    def normalize(self, column: pd.Series) -> pd.DataFrame:
        if isinstance(column[0], str):
            column = column.str[-64:].astype("int")

        if self.truncate:
            column = column % (2 ** self.num_bits)

        bits_dataframe = pd.DataFrame(
            (column[:, None] & (1 << np.arange(self.num_bits))) > 0
        )

        final_dataframe = pd.DataFrame()
        for column in bits_dataframe.columns[::-1]:
            final_dataframe[f"{column}_0"] = (~bits_dataframe[column]).astype("int")
            final_dataframe[f"{column}_1"] = bits_dataframe[column].astype("int")

        return final_dataframe

    def denormalize(self, bin_x):
        if len(bin_x.shape) == 3:
            # This is a timeseries field
            a, b, c = bin_x.shape
            if self.num_bits * 2 != c:
                raise ValueError(
                    f"Dimension is {c}. Expected dimension is {self.num_bits * 2}"
                )
            return self.denormalize(bin_x.reshape(a * b, c)).to_numpy().reshape(a, b)
        df_bin = pd.DataFrame(bin_x)
        chosen_bits = (df_bin > df_bin.shift(axis=1)).drop(
            range(0, self.num_bits * 2, 2), axis=1
        )
        return chosen_bits.dot(1 << np.arange(self.num_bits - 1, -1, -1))

    def get_output_type(self):
        outputs = []

        for i in range(self.num_bits):
            outputs.append(Output(type_=OutputType.DISCRETE, dim=2))

        return outputs

    def get_output_dim(self) -> int:
        return 2 * self.num_bits  # type: ignore


class Word2VecField(Field):
    def __init__(self, word2vec_size, word2vec_cols, *args, **kwargs):
        super(Word2VecField, self).__init__(*args, **kwargs)

        self.word2vec_size = word2vec_size
        self.norm_option = Normalization.MINUSONE_ONE
        self.dict_encoding_type_vs_cols = get_word2vec_type_col(word2vec_cols)

    def normalize(self, x, embed_model):
        return np.array(
            [get_vector(embed_model, str(xi), norm_option=True) for xi in x]
        )

    def _denormalize(self, norm_x, encoding_type, dict_annoyIndex, dict_annDictPair):
        # When constructing the dict_annDictPair, dict_annDictPair[encoding_type]: dict{}
        # is a k v pair where k is an integer. After using Json.save() then Json.load(), it
        # will change k into str. Therefore, when we are using dict_annDictPair[encoding_type]
        # the k should be casted to int.
        return get_original_objs(
            dict_annoyIndex[encoding_type],
            norm_x,
            {int(k): v for k, v in dict_annDictPair[encoding_type].items()},
        )

    def denormalize(self, norm_x):
        with open(get_annoy_dict_idx_ele_for_word2vec(), "r") as readfile:
            dict_annDictPair = json.load(readfile)
        dict_annoyIndex = {}
        for encoding_type in dict_annDictPair:
            type_ann = AnnoyIndex(self.word2vec_size, "angular")
            type_ann.load(get_annoyIndex_for_word2vec(encoding_type))
            dict_annoyIndex[encoding_type] = type_ann

        for k in self.dict_encoding_type_vs_cols:
            if self.name in self.dict_encoding_type_vs_cols[k]:
                encoding_type = k
                break
        else:
            raise ValueError("Cannot find the word2vec key!")

        if len(norm_x.shape) == 3:
            # This is a timeseries field
            return np.array(
                [
                    self._denormalize(
                        x, encoding_type, dict_annoyIndex, dict_annDictPair
                    )
                    for x in norm_x
                ]
            )
        return np.asarray(
            self._denormalize(norm_x, encoding_type, dict_annoyIndex, dict_annDictPair)
        )

    def get_output_type(self):
        return Output(
            type_=OutputType.CONTINUOUS,
            dim=self.get_output_dim(),
            normalization=self.norm_option,
        )

    def get_output_dim(self) -> int:
        return self.word2vec_size  # type: ignore


def field_config_to_key(field: Config) -> FieldKey:
    return (
        field.get("name") or field.get("column") or str(field.columns),
        field.type,
        field.get("encoding", ""),
    )


def key_from_field(field: Field) -> FieldKey:
    return str(field.name), field.__class__.__name__, ""
