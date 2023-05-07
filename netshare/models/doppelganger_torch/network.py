import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from netshare.utils import Output, OutputType, Normalization


class AttrDiscriminator(torch.nn.Module):
    def __init__(
        self,
        input_attribute_dim,
        num_layers=5,
        num_units=200,
        scope_name="attr_discriminator",
        *args,
        **kwargs
    ):
        super(AttrDiscriminator, self).__init__()
        self.scope_name = scope_name

        layers = []
        layer_num_units = input_attribute_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(layer_num_units, num_units))
            layers.append(torch.nn.ReLU())
            layer_num_units = num_units

        layers.append(torch.nn.Linear(layer_num_units, 1))
        self.attr_disc = torch.nn.Sequential(*layers)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():

            if "weight" in name:
                torch.nn.init.xavier_uniform_(p.data)
            elif "bias" in name:
                p.data.fill_(0)

    def forward(self, input_attribute):
        input_ = torch.flatten(input_attribute, start_dim=1)

        return self.attr_disc(input_)


class Discriminator(torch.nn.Module):
    def __init__(
        self,
        max_sequence_len,
        input_feature_dim,
        input_attribute_dim,
        num_layers=5,
        num_units=200,
        scope_name="discriminator",
        *args,
        **kwargs
    ):
        super(Discriminator, self).__init__()
        self.scope_name = scope_name

        input_dim = max_sequence_len * input_feature_dim + input_attribute_dim

        layers = []
        layer_num_units = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(layer_num_units, num_units))
            layers.append(torch.nn.ReLU())
            layer_num_units = num_units

        layers.append(torch.nn.Linear(layer_num_units, 1))
        self.disc = torch.nn.Sequential(*layers)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_uniform_(p.data)
            elif "bias" in name:
                p.data.fill_(0)

    def forward(self, input_feature, input_attribute):
        input_feature = torch.flatten(input_feature, start_dim=1, end_dim=2)
        input_attribute = torch.flatten(input_attribute, start_dim=1)
        input_ = torch.cat((input_feature, input_attribute), dim=1)
        return self.disc(input_)


class DoppelGANgerGenerator(torch.nn.Module):
    def __init__(
        self,
        attr_latent_dim,
        feature_latent_dim,
        feature_outputs,
        attribute_outputs,
        real_attribute_mask,
        sample_len,
        attribute_num_units=100,
        attribute_num_layers=3,
        feature_num_units=100,
        feature_num_layers=1,
        batch_size=100,
        use_adaptive_rolling=True,
        device="cpu",
        scope_name="doppelganger_generator",
        *args,
        **kwargs
    ):
        super(DoppelGANgerGenerator, self).__init__()
        self.scope_name = scope_name
        self.device = device

        self.feature_out_dim = np.sum([t.dim for t in feature_outputs])
        self.feature_outputs = feature_outputs
        self.attribute_out_dim = np.sum([t.dim for t in attribute_outputs])
        self.attribute_outputs = attribute_outputs
        self.feature_num_layers = feature_num_layers
        self.feature_num_units = feature_num_units
        self.batch_size = batch_size
        self.use_adaptive_rolling = use_adaptive_rolling

        self.use_addi_attribute_generator = True

        self.real_attribute_outputs = []
        self.addi_attribute_outputs = []
        self.real_attribute_out_dim = 0
        self.addi_attribute_out_dim = 0
        for i in range(len(attribute_outputs)):
            if real_attribute_mask[i]:
                self.real_attribute_outputs.append(self.attribute_outputs[i])
                self.real_attribute_out_dim += attribute_outputs[i].dim
            else:
                self.addi_attribute_outputs.append(self.attribute_outputs[i])
                self.addi_attribute_out_dim += attribute_outputs[i].dim

        if len(self.addi_attribute_outputs) == 0:
            self.use_addi_attribute_generator = False

        for i in range(len(real_attribute_mask) - 1):
            if real_attribute_mask[i] == False and real_attribute_mask[i + 1] == True:
                raise Exception("Real attribute should come first")

        # Define generator without the output layer for real attributes
        layers = []
        num_layer_units = attr_latent_dim
        for i in range(attribute_num_layers - 1):
            layers.append(torch.nn.Linear(num_layer_units, attribute_num_units))
            layers.append(torch.nn.ReLU())
            layers.append(
                torch.nn.BatchNorm1d(
                    num_features=attribute_num_units, eps=1e-5, momentum=0.9
                )
            )
            num_layer_units = attribute_num_units

        self.real_attribute_gen_without_last_layer = torch.nn.Sequential(
            *layers)

        # Define the output layer of the real attributes generator
        self.real_attribute_gen_last_layer = torch.nn.ModuleList()
        for i in range(len(self.real_attribute_outputs)):
            attr_out_layer = [
                torch.nn.Linear(
                    num_layer_units, self.real_attribute_outputs[i].dim)]
            if self.real_attribute_outputs[i].type_ == OutputType.DISCRETE:
                attr_out_layer.append(torch.nn.Softmax(dim=-1))
            else:
                if (
                    self.real_attribute_outputs[i].normalization
                    == Normalization.ZERO_ONE
                ):
                    attr_out_layer.append(torch.nn.Sigmoid())
                else:
                    attr_out_layer.append(torch.nn.Tanh())
            self.real_attribute_gen_last_layer.append(
                torch.nn.Sequential(*attr_out_layer)
            )

        if self.use_addi_attribute_generator:
            # Define generator without the output layer for addi attributes
            layers = []
            num_layer_units = attr_latent_dim + self.real_attribute_out_dim
            for i in range(attribute_num_layers - 1):
                layers.append(torch.nn.Linear(
                    num_layer_units, attribute_num_units))
                layers.append(torch.nn.ReLU())
                layers.append(
                    torch.nn.BatchNorm1d(
                        num_features=attribute_num_units, eps=1e-5, momentum=0.9
                    )
                )
                num_layer_units = attribute_num_units

            self.addi_attribute_gen_without_last_layer = torch.nn.Sequential(
                *layers)

            # Define the output layer of the addi attributes generator
            self.addi_attribute_gen_last_layer = torch.nn.ModuleList()
            for i in range(len(self.addi_attribute_outputs)):
                attr_out_layer = [
                    torch.nn.Linear(
                        num_layer_units, self.addi_attribute_outputs[i].dim)]
                if self.addi_attribute_outputs[i].type_ == OutputType.DISCRETE:
                    attr_out_layer.append(torch.nn.Softmax(dim=-1))
                else:
                    if (
                        self.addi_attribute_outputs[i].normalization
                        == Normalization.ZERO_ONE
                    ):
                        attr_out_layer.append(torch.nn.Sigmoid())
                    else:
                        attr_out_layer.append(torch.nn.Tanh())
                self.addi_attribute_gen_last_layer.append(
                    torch.nn.Sequential(*attr_out_layer)
                )

        # Define the feature generator
        self.lstm_module = torch.nn.LSTM(
            self.real_attribute_out_dim
            + self.addi_attribute_out_dim
            + feature_latent_dim,
            self.feature_num_units,
            self.feature_num_layers,
            batch_first=True,
        )

        self.feature_gen_last_layer = torch.nn.ModuleList()
        feature_len = len(self.feature_outputs)
        num_layer_units = self.feature_num_units
        for i in range(feature_len * sample_len):

            feature_out_layer = [
                torch.nn.Linear(
                    num_layer_units, self.feature_outputs[i % feature_len].dim
                )
            ]
            if self.feature_outputs[i % feature_len].type_ == OutputType.DISCRETE:
                feature_out_layer.append(torch.nn.Softmax(dim=-1))
            else:
                if (
                    self.feature_outputs[i % feature_len].normalization
                    == Normalization.ZERO_ONE
                ):
                    feature_out_layer.append(torch.nn.Sigmoid())
                else:
                    feature_out_layer.append(torch.nn.Tanh())

            self.feature_gen_last_layer.append(
                torch.nn.Sequential(*feature_out_layer))

        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4): (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)
            elif "linear" in name:
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(p.data)
                elif "bias" in name:
                    p.data.fill_(0)

    def forward(
        self,
        real_attribute_noise,
        addi_attribute_noise,
        feature_input_noise,
        h0,
        c0,
        given_attribute=None,
        given_attribute_discrete=None,
    ):
        # Shape of feature_input_noise is [batch_size, sample_time, latent_dim]
        # Shape of real_attribute_noise is [batch_size, latent_dim]
        # Shape of addi_attribute_noise is [batch_size, latent_dim]

        if given_attribute is None and given_attribute_discrete is None:

            attribute = []
            attribute_discrete = []

            # Calculate the forwarding of attribute generator
            real_attribute = []
            real_attribute_discrete = []
            real_attribute_gen_tmp = self.real_attribute_gen_without_last_layer(
                real_attribute_noise
            )

            for attr_layer in self.real_attribute_gen_last_layer:
                attr_sub_output = attr_layer(real_attribute_gen_tmp)
                if isinstance(attr_layer[-1], torch.nn.Softmax):
                    attr_sub_output_discrete = F.one_hot(
                        torch.argmax(attr_sub_output, dim=1),
                        num_classes=attr_sub_output.shape[1],
                    )
                else:
                    attr_sub_output_discrete = attr_sub_output
                real_attribute.append(attr_sub_output)
                real_attribute_discrete.append(attr_sub_output_discrete)
            real_attribute = torch.cat(real_attribute, dim=1)
            real_attribute_discrete = torch.cat(
                real_attribute_discrete, dim=1).detach()

            attribute.append(real_attribute)
            attribute_discrete.append(real_attribute_discrete)

            if self.use_addi_attribute_generator:
                # Calculate the forwarding of the addi attribute generator
                addi_attribute_input = torch.cat(
                    (real_attribute_discrete, addi_attribute_noise), dim=1
                )
                addi_attribute = []
                addi_attribute_discrete = []
                addi_attribute_gen_tmp = self.addi_attribute_gen_without_last_layer(
                    addi_attribute_input)

                for attr_layer in self.addi_attribute_gen_last_layer:
                    addi_attr_sub_output = attr_layer(addi_attribute_gen_tmp)
                    if isinstance(attr_layer[-1], torch.nn.Softmax):
                        addi_attr_sub_output_discrete = F.one_hot(
                            torch.argmax(addi_attr_sub_output, dim=1),
                            num_classes=attr_sub_output.shape[1],
                        )
                    else:
                        addi_attr_sub_output_discrete = addi_attr_sub_output
                    addi_attribute.append(addi_attr_sub_output)
                    addi_attribute_discrete.append(
                        addi_attr_sub_output_discrete)
                addi_attribute = torch.cat(addi_attribute, dim=1)
                addi_attribute_discrete = torch.cat(
                    addi_attribute_discrete, dim=1
                ).detach()

                attribute.append(addi_attribute)
                attribute_discrete.append(addi_attribute_discrete)

            attribute = torch.cat(attribute, dim=1)
            attribute_discrete = torch.cat(attribute_discrete, dim=1)

        else:
            attribute = given_attribute
            attribute_discrete = given_attribute_discrete

        # Calculate the forwarding of the feature generator
        attribute_ = torch.unsqueeze(attribute_discrete, dim=1)
        feature_input_ = attribute_.expand(
            -1, feature_input_noise.shape[1], -1
        ).detach()
        feature_input = torch.cat((feature_input_, feature_input_noise), dim=2)

        ##########
        if self.use_adaptive_rolling:
            hn, cn = h0, c0
            feature = []
            batch_size = feature_input.size()[0]
            steps = feature_input.size()[1]
            data = feature_input.unbind(1)
            curr_step = 0
            for xt in data:
                output_per_step, (hn, cn) = self.lstm_module(
                    xt[:, None, :], (hn, cn))
                feature_per_step = []
                for feature_layer in self.feature_gen_last_layer:
                    feature_sub_output = feature_layer(output_per_step)
                    feature_per_step.append(feature_sub_output)

                feature_per_step = torch.cat(feature_per_step, dim=2)
                gen_flag_per_step = feature_per_step[
                    :, :, self.feature_out_dim - 2:: self.feature_out_dim
                ]
                feature.append(feature_per_step)
                curr_step += 1
                tmp_, _ = torch.min((gen_flag_per_step > 0.5).int(), 2)
                if torch.max(tmp_) == 0:
                    # All the gen flag is false in this case
                    break

            _zeros = torch.zeros(
                batch_size, (steps - curr_step), feature[0].size()[2] - 1
            ).to(self.device)
            _ones = torch.ones(
                batch_size, (steps - curr_step),
                1).to(
                self.device)
            feature.append(torch.cat((_zeros, _ones), 2))
            feature = torch.cat(feature, dim=1)
        else:
            feature_rnn_output_tmp, _ = self.lstm_module(
                feature_input, (h0, c0)
            )

            feature = []
            for feature_layer in self.feature_gen_last_layer:
                feature_sub_output = feature_layer(feature_rnn_output_tmp)
                feature.append(feature_sub_output)
            feature = torch.cat(feature, dim=2)
        ###########

        # feature: (batch_size, step/sample_len, num_feature*sample_len)
        all_gen_flag = feature[
            :, :, self.feature_out_dim - 2:: self.feature_out_dim
        ]

        # mask shape: (batch_size, step/sample_len)
        mask_, _ = torch.min((all_gen_flag > 0.5).int(), 2)

        if mask_.size()[1] > 1:
            tmp_mask = mask_[:, 0]
            for i in range(1, mask_.size()[1]):
                mask_[:, i] *= tmp_mask
                tmp_mask = mask_[:, i]

        mask_shift = torch.cat((torch.ones(mask_.size()[0], 1).to(self.device), mask_[:, :-1]), axis=1)
        # mask shape: (batch_size, step/sample_len, 1)
        mask_shift = torch.unsqueeze(mask_shift, 2)
        # mask shape: (batch_size, step/sample_len, num_feature*sample_len)
        mask_shift = mask_shift.expand(feature.shape[0], feature.shape[1], feature.shape[2])
        feature *= mask_shift

        feature = torch.reshape(
            feature,
            (
                feature.shape[0],
                int(feature.shape[1] * feature.shape[2] / self.feature_out_dim),
                self.feature_out_dim,
            ),
        )

        return attribute, attribute_discrete, feature
