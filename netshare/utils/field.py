import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from collections import defaultdict
from annoy import AnnoyIndex

from .output import Normalization, OutputType, Output
from ..pre_post_processors.netshare.embedding_helper import get_vector, get_original_obj, get_original_objs

EPS = 1e-8


class Field(object):
    def __init__(self, name):
        self.name = name

    def normalize(self):
        raise NotImplementedError

    def denormalize(self):
        raise NotImplementedError

    def getOutputType(self):
        raise NotImplementedError


class ContinuousField(Field):
    def __init__(
            self, norm_option, min_x=None, max_x=None, dim_x=1,
            log1p_norm=False, *args, **kwargs):
        super(ContinuousField, self).__init__(*args, **kwargs)

        self.min_x = min_x
        self.max_x = max_x
        self.norm_option = norm_option
        self.dim_x = dim_x
        self.log1p_norm = log1p_norm
        if self.log1p_norm:
            self.min_x = np.log1p(self.min_x)
            self.max_x = np.log1p(self.max_x)

    # Normalize x in [a, b]: x' = (b-a)(x-min x)/(max x - minx) + a
    def normalize(self, x):
        if x.shape[-1] != self.dim_x:
            raise ValueError(f"Dimension is {x.shape[-1]}. "
                             f"Expected dimension is {self.dim_x}")
        if self.log1p_norm:
            x = np.log1p(x)

        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            return np.asarray((x - self.min_x) / (self.max_x - self.min_x))

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            return np.asarray(2 * (x - self.min_x)
                              / (self.max_x - self.min_x) - 1)
        else:
            raise Exception("Not valid normalization option!")

    def denormalize(self, norm_x):
        if norm_x.shape[-1] != self.dim_x:
            raise ValueError(f"Dimension is {norm_x.shape[-1]}. "
                             f"Expected dimension is {self.dim_x}")
        norm_x = norm_x.astype(np.float64)  # Convert to float64 for precision

        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            to_return = norm_x * float(self.max_x - self.min_x) + self.min_x

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            to_return = (norm_x + 1) / 2.0 * \
                float(self.max_x - self.min_x) + self.min_x

        else:
            raise Exception("Not valid normalization option!")

        if self.log1p_norm:
            to_return = np.expm1(to_return)

        return to_return

    def getOutputType(self):
        return Output(
            type_=OutputType.CONTINUOUS,
            dim=self.dim_x,
            normalization=self.norm_option
        )


class DiscreteField(Field):
    def __init__(self, choices, *args, **kwargs):
        super(DiscreteField, self).__init__(*args, **kwargs)

        if not isinstance(choices, list):
            raise Exception("choices should be a list")
        self.choices = choices
        self.dim_x = len(choices)

    def normalize(self, x):
        if not isinstance(x, (list, np.ndarray)):
            norm_x = [x]
        else:
            norm_x = x
        norm_x = pd.DataFrame(norm_x).astype(
            pd.CategoricalDtype(categories=self.choices))
        norm_x = pd.get_dummies(norm_x).to_numpy()
        if not isinstance(x, (list, np.ndarray)):
            norm_x = norm_x[0]

        return norm_x

    def denormalize(self, norm_x):
        index = np.argmax(norm_x, axis=-1)

        return np.asarray(self.choices)[index]

    def getOutputType(self):
        return Output(
            type_=OutputType.DISCRETE,
            dim=len(self.choices)
        )


class BitField(Field):
    def __init__(self, num_bits, *args, **kwargs):
        super(BitField, self).__init__(*args, **kwargs)

        self.num_bits = num_bits
        self.dim_x = 2*num_bits

    def normalize(self, decimal_x):
        bin_x = bin(int(decimal_x))[2:].zfill(self.num_bits)
        bin_x = [int(b) for b in bin_x]

        bits = []
        for b in bin_x:
            if b == 0:
                bits += [1.0, 0.0]

            elif b == 1:
                bits += [0.0, 1.0]

            else:
                print("Binary number is zero or one!")

        return bits

    def denormalize(self, bin_x):
        if len(bin_x.shape) == 3:
            # This is a timeseries field
            a, b, c = bin_x.shape
            if self.num_bits * 2 != c:
                raise ValueError(
                    f"Dimension is {c}. Expected dimension is {self.num_bits * 2}"
                )
            return self.denormalize(
                bin_x.reshape(a * b, c)).to_numpy().reshape(
                a, b)
        df_bin = pd.DataFrame(bin_x)
        chosen_bits = (df_bin > df_bin.shift(axis=1)).drop(
            range(0, self.num_bits * 2, 2), axis=1
        )
        return chosen_bits.dot(1 << np.arange(self.num_bits - 1, -1, -1))

    def getOutputType(self):
        outputs = []

        for i in range(self.num_bits):
            outputs.append(Output(type_=OutputType.DISCRETE, dim=2))

        return outputs


class Word2VecField(Field):
    def __init__(
            self, word2vec_size, pre_processed_data_folder, word2vec_type, *
            args, **kwargs):
        super(Word2VecField, self).__init__(*args, **kwargs)

        self.word2vec_size = word2vec_size
        self.preprocessed_data_folder = pre_processed_data_folder
        self.word2vec_type = word2vec_type
        self.dim_x = word2vec_size
        self.norm_option = Normalization.MINUSONE_ONE

    def normalize(self, x, embed_model):
        return np.array(
            [get_vector(embed_model, str(xi), norm_option=True) for xi in x]
        )

    def denormalize(self, norm_x):
        # load Annoy and Dict
        type_ann = AnnoyIndex(self.word2vec_size, 'angular')
        type_ann.load(os.path.join(
            self.preprocessed_data_folder,
            f"{self.word2vec_type}_ann.ann"))
        with open(os.path.join(self.preprocessed_data_folder, f"{self.word2vec_type}_dict.json"), 'r') as f:
            type_dict = json.load(f)

        if len(norm_x.shape) == 3:
            # This is a timeseries field
            return np.array(
                [
                    get_original_objs(
                        ann=type_ann,
                        vectors=x,
                        dic={int(k): v for k, v in type_dict.items()}
                    )
                    for x in norm_x
                ]
            )
        return np.asarray(get_original_objs(
            ann=type_ann,
            vectors=norm_x,
            dic={int(k): v for k, v in type_dict.items()}
        ))

    def getOutputType(self):
        return Output(
            type_=OutputType.CONTINUOUS,
            dim=self.dim_x,
            normalization=self.norm_option
        )
