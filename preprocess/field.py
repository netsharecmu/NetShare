import numpy as np
from output import *

class Field(object):
    def __init__(self, name):
        self.name = name

    def normalize(self):
        raise NotImplementedError

    def denormalize(self):
        raise NotImplementedError

    def getOutputType(self):
        raise NotImplementedError

# Normalize x in [a, b]: x' = (b-a)(x-min x)/(max x - minx) + a
class ContinuousField(Field):
    def __init__(self, norm_option, min_x=None, max_x=None, dim_x=1, *args, **kwargs):
        super(ContinuousField, self).__init__(*args, **kwargs)

        self.min_x = min_x
        self.max_x = max_x
        self.norm_option = norm_option
        self.dim_x = dim_x

    def normalize(self, x):
        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            return np.asarray((x - self.min_x) / (self.max_x - self.min_x))

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            return np.asarray(2 * (x - self.min_x) / (self.max_x - self.min_x) - 1)

        else:
            raise Exception("Not valid normalization option!")

    def denormalize(self, norm_x):
        # [0, 1] normalization
        if self.norm_option == Normalization.ZERO_ONE:
            return norm_x * float(self.max_x - self.min_x) + self.min_x

        # [-1, 1] normalization
        elif self.norm_option == Normalization.MINUSONE_ONE:
            return (norm_x+1) / 2.0 * float(self.max_x - self.min_x) + self.min_x

        else:
            raise Exception("Not valid normalization option!")

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

    def normalize(self, x):
        index = self.choices.index(x)

        norm_x = np.zeros_like(self.choices, dtype=float)
        norm_x[index] = 1.0

        return list(norm_x)

    def denormalize(self, norm_x):
        index = np.argmax(norm_x)

        return self.choices[index]

    def getOutputType(self):
        return Output(
            type_=OutputType.DISCRETE,
            dim=len(self.choices)
        )

class BitField(Field):
    def __init__(self, num_bits, *args, **kwargs):
        super(BitField, self).__init__(*args, **kwargs)

        self.num_bits = num_bits

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
        if not isinstance(bin_x, list):
            raise Exception("Bit array should be a list")

        assert len(bin_x) == 2*self.num_bits, "length of bit array is wrong!"

        bits = "0b"
        for i in range(self.num_bits):
            index = np.argmax(bin_x[2*i:2*(i+1)])

            if index == 0:
                bits += "0"

            elif index == 1:
                bits += "1"

            else:
                raise Exception("Bits array is ZERO or ONE!")

        decimal_x = int(bits, 2) 

        return decimal_x


    def getOutputType(self):
        outputs = []

        for i in range(self.num_bits):
            outputs.append(
                Output(
                    type_=OutputType.DISCRETE,
                    dim=2
                ))

        return outputs