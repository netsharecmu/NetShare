import os
import sys
import torch
import datetime
import numpy as np
from tqdm import tqdm
from .network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

try:
    from opacus.optimizers import DPOptimizer
    from opacus.accountants import RDPAccountant
    from opacus import GradSampleModule
    from .privacy_util import compute_dp_sgd_privacy
except BaseException:
    pass


class DoppelGANger(object):
    def __init__(
        self,
        # General training related parameters
        checkpoint_dir,
        sample_dir,
        time_path,
        batch_size,
        real_attribute_mask,
        max_sequence_len,
        sample_len,
        data_feature_outputs,
        data_attribute_outputs,
        vis_freq,
        vis_num_sample,
        d_rounds,
        g_rounds,
        d_gp_coe,
        num_packing,
        use_attr_discriminator,
        attr_d_gp_coe,
        g_attr_d_coe,
        epoch_checkpoint_freq,
        attribute_latent_dim,
        feature_latent_dim,
        g_lr,
        g_beta1,
        d_lr,
        d_beta1,
        attr_d_lr,
        attr_d_beta1,
        adam_eps,
        adam_amsgrad,
        # DoppelGANgerGenerator related hyper-parameters
        generator_attribute_num_units,
        generator_attribute_num_layers,
        generator_feature_num_units,
        generator_feature_num_layers,
        use_adaptive_rolling,
        # Discriminator related hyper-parameters
        discriminator_num_layers,
        discriminator_num_units,
        # Attr discriminator related hyper-parameters
        # Please ignore these params if use_attr_discriminator = False
        attr_discriminator_num_layers,
        attr_discriminator_num_units,
        # Pretrain-related
        restore=False,
        pretrain_dir=None
    ):

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.time_path = time_path
        self.batch_size = batch_size
        self.real_attribute_mask = real_attribute_mask
        self.max_sequence_len = max_sequence_len
        self.sample_len = sample_len
        self.data_feature_outputs = data_feature_outputs
        self.data_attribute_outputs = data_attribute_outputs
        self.vis_freq = vis_freq
        self.vis_num_sample = vis_num_sample
        self.num_packing = num_packing
        self.use_attr_discriminator = use_attr_discriminator
        self.d_rounds = d_rounds
        self.g_rounds = g_rounds
        self.d_gp_coe = d_gp_coe
        self.attr_d_gp_coe = attr_d_gp_coe
        self.g_attr_d_coe = g_attr_d_coe
        self.epoch_checkpoint_freq = epoch_checkpoint_freq
        self.attribute_latent_dim = attribute_latent_dim
        self.feature_latent_dim = feature_latent_dim
        self.g_lr = g_lr
        self.g_beta1 = g_beta1
        self.d_lr = d_lr
        self.d_beta1 = d_beta1
        self.attr_d_lr = attr_d_lr
        self.attr_d_beta1 = attr_d_beta1
        self.adam_eps = adam_eps
        self.adam_amsgrad = adam_amsgrad

        self.generator_attribute_num_units = generator_attribute_num_units
        self.generator_attribute_num_layers = generator_attribute_num_layers
        self.generator_feature_num_units = generator_feature_num_units
        self.generator_feature_num_layers = generator_feature_num_layers
        self.use_adaptive_rolling = use_adaptive_rolling

        self.discriminator_num_layers = discriminator_num_layers
        self.discriminator_num_units = discriminator_num_units

        self.attr_discriminator_num_layers = attr_discriminator_num_layers
        self.attr_discriminator_num_units = attr_discriminator_num_units

        self.restore = restore
        self.pretrain_dir = pretrain_dir

        self.EPS = 1e-8

        self.MODEL_NAME = "model"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.max_sequence_len % self.sample_len != 0:
            raise Exception("length must be a multiple of sample_len")

        self.sample_time = self.max_sequence_len / self.sample_len

        self.is_build = False

        self.feature_dim = np.sum([t.dim for t in self.data_feature_outputs])
        self.attribute_dim = np.sum(
            [t.dim for t in self.data_attribute_outputs])
        self._build()

    def check_data(self):
        self.gen_flag_dims = []

        dim = 0
        for output in self.data_feature_outputs:
            if output.is_gen_flag:
                if output.dim != 2:
                    raise Exception("gen flag output's dim should be 2")
                self.gen_flag_dims = [dim, dim + 1]
                break
            dim += output.dim
        if len(self.gen_flag_dims) == 0:
            raise Exception("gen flag not found")

        if self.data_feature.shape[2] != np.sum(
            [t.dim for t in self.data_feature_outputs]
        ):
            raise Exception(
                "feature dimension does not match data_feature_outputs")

        if len(self.data_gen_flag.shape) != 2:
            raise Exception("data_gen_flag should be 2 dimension")

        self.data_gen_flag = np.expand_dims(self.data_gen_flag, 2)

    def train(self, epochs, data_feature, data_attribute, data_gen_flag):
        self.epochs = epochs
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.data_gen_flag = data_gen_flag

        log_dir = f"{self.checkpoint_dir}/runs"
        self.writer = SummaryWriter(log_dir=log_dir)

        self.check_data()

        dataset = TensorDataset(
            torch.Tensor(data_attribute), torch.Tensor(data_feature)
        )

        self._train(dataset)

    def generate(
        self,
        num_samples,
        given_attribute=None,
        given_attribute_discrete=None,
        return_gen_flag_feature=False,
    ):
        if self.is_build == False:
            raise Exception("model has not been trained")

        if num_samples is not None:
            num_batches = num_samples // self.batch_size
        if num_samples % self.batch_size != 0:
            num_batches += 1

        real_attribute_noise = self._gen_attribute_input_noise(num_samples).to(
            self.device
        )
        addi_attribute_noise = self._gen_attribute_input_noise(num_samples).to(
            self.device
        )
        feature_input_noise = self._gen_feature_input_noise(
            num_samples, self.sample_time
        ).to(self.device)
        h0 = Variable(
            torch.normal(
                0, 1, (self.generator.feature_num_layers, num_samples, self.generator.feature_num_units)
            )).to(self.device)
        c0 = Variable(
            torch.normal(
                0, 1, (self.generator.feature_num_layers, num_samples, self.generator.feature_num_units)
            )).to(self.device)

        generated_data_list = []
        for n_batch in range(num_batches):
            if given_attribute is not None and given_attribute is not None:
                batch_given_attribute = given_attribute[
                    n_batch * self.batch_size: (n_batch + 1) * self.batch_size
                ]
                batch_given_attribute_discrete = given_attribute_discrete[
                    n_batch * self.batch_size: (n_batch + 1) * self.batch_size
                ]
            else:
                batch_given_attribute = None
                batch_given_attribute_discrete = None

            generated_data_list.append(
                self._generate(
                    real_attribute_noise=real_attribute_noise
                    [n_batch * self.batch_size: (n_batch + 1) * self.batch_size],
                    addi_attribute_noise=addi_attribute_noise
                    [n_batch * self.batch_size: (n_batch + 1) * self.batch_size],
                    feature_input_noise=feature_input_noise
                    [n_batch * self.batch_size: (n_batch + 1) * self.batch_size],
                    h0=h0
                    [:, n_batch * self.batch_size: (n_batch + 1) * self.
                     batch_size, :],
                    c0=c0
                    [:, n_batch * self.batch_size: (n_batch + 1) * self.
                     batch_size, :],
                    given_attribute=batch_given_attribute,
                    given_attribute_discrete=batch_given_attribute_discrete))

        attribute, attribute_discrete, feature = tuple(
            np.concatenate(d, axis=0) for d in zip(*generated_data_list)
        )

        gen_flag = np.array(np.round(feature[:, :, -2]))

        # if for a session, all the gen_flag are 1, np.argmin will return 0. Therefore, add a new "0"
        # at the end of each session to solve this edge case. Make sure np.argmin will return the first
        # index of "0"
        gen_flag_helper = np.concatenate(
            [gen_flag, np.zeros((gen_flag.shape[0], 1))], axis=1
        )
        min_indicator = np.argmin(gen_flag_helper, axis=1)
        for row, min_idx in enumerate(min_indicator):
            gen_flag[row, min_idx:] = 0.0
        lengths = np.sum(gen_flag, axis=1)

        if not return_gen_flag_feature:
            feature = feature[:, :, :-2]
        return feature, attribute, attribute_discrete, gen_flag, lengths

    def save(self, file_path, only_generator=False, include_optimizer=False):
        if only_generator:
            state = {
                "generator_state_dict": self.generator.state_dict(),
            }
        else:
            state = {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
            }
            if self.use_attr_discriminator:
                state[
                    "attr_discriminator_state_dict"
                ] = self.attr_discriminator.state_dict()

            if include_optimizer:
                state[
                    "generator_optimizer_state_dict"
                ] = self.opt_generator.state_dict()
                state[
                    "discriminator_optimizer_state_dict"
                ] = self.opt_discriminator.state_dict()

                if self.use_attr_discriminator:
                    state[
                        "attr_discriminator_optimizer_state_dict"
                    ] = self.opt_attr_discriminator.state_dict()
        torch.save(state, file_path)

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Directory to load pytorch model doesn't exist")

        state = torch.load(model_path)
        self.generator.load_state_dict(state["generator_state_dict"])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])

        if self.use_attr_discriminator:
            self.attr_discriminator.load_state_dict(
                state["attr_discriminator_state_dict"]
            )

        if "generator_optimizer_state_dict" in state:
            self.opt_generator.load_state_dict(
                state["generator_optimizer_state_dict"])
            self.opt_discriminator.load_state_dict(
                state["discriminator_optimizer_state_dict"]
            )

            if self.use_attr_discriminator:
                self.opt_attr_discriminator.load_state_dict(
                    state["attr_discriminator_optimizer_state_dict"]
                )

    def _build(self):
        self.generator = DoppelGANgerGenerator(
            attr_latent_dim=self.attribute_latent_dim,
            feature_latent_dim=self.feature_latent_dim,
            feature_outputs=self.data_feature_outputs,
            attribute_outputs=self.data_attribute_outputs,
            real_attribute_mask=self.real_attribute_mask,
            sample_len=self.sample_len,
            attribute_num_units=self.generator_attribute_num_units,
            attribute_num_layers=self.generator_attribute_num_layers,
            feature_num_units=self.generator_feature_num_units,
            feature_num_layers=self.generator_feature_num_layers,
            batch_size=self.batch_size,
            use_adaptive_rolling=self.use_adaptive_rolling,
            device=self.device
        )
        self.generator.to(self.device)

        self.discriminator = Discriminator(
            max_sequence_len=self.max_sequence_len,
            input_feature_dim=self.feature_dim * self.num_packing,
            input_attribute_dim=self.attribute_dim * self.num_packing,
            num_layers=self.discriminator_num_layers,
            num_units=self.discriminator_num_units,
        )
        self.discriminator.to(self.device)

        if self.use_attr_discriminator:
            self.attr_discriminator = AttrDiscriminator(
                input_attribute_dim=self.attribute_dim * self.num_packing,
                num_layers=self.attr_discriminator_num_layers,
                num_units=self.attr_discriminator_num_units,
            )
            self.attr_discriminator.to(self.device)

        self.opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.d_lr,
            betas=(self.d_beta1, 0.999),
            eps=self.adam_eps,
            amsgrad=self.adam_amsgrad,
        )

        if self.use_attr_discriminator:
            self.opt_attr_discriminator = torch.optim.Adam(
                self.attr_discriminator.parameters(),
                lr=self.attr_d_lr,
                betas=(self.attr_d_beta1, 0.999),
                eps=self.adam_eps,
                amsgrad=self.adam_amsgrad,
            )
        else:
            self.opt_attr_discriminator = None

        self.opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(self.g_beta1, 0.999),
            eps=self.adam_eps,
            amsgrad=self.adam_amsgrad,
        )

        self.is_build = True

    def _gen_attribute_input_noise(self, batch_size):
        return torch.randn(
            size=[int(batch_size),
                  int(self.attribute_latent_dim)])

    def _gen_feature_input_noise(self, batch_size, length):
        return torch.randn(
            size=[int(batch_size), int(length), int(self.feature_latent_dim)]
        )

    def _calculate_gp_dis(
            self, batch_size, fake_feature, real_feature, fake_attribute,
            real_attribute):

        alpha_dim2 = torch.FloatTensor(
            batch_size, 1).uniform_(1).to(
            self.device)
        alpha_dim3 = torch.unsqueeze(alpha_dim2, 2).to(self.device)
        differences_input_feature = fake_feature - real_feature
        interpolates_input_feature = (
            real_feature + alpha_dim3 * differences_input_feature
        )
        differences_input_attribute = fake_attribute - real_attribute
        interpolates_input_attribute = real_attribute + (
            alpha_dim2 * differences_input_attribute
        )
        mixed_scores = self.discriminator(
            interpolates_input_feature, interpolates_input_attribute
        )
        gradients = torch.autograd.grad(
            inputs=[interpolates_input_feature, interpolates_input_attribute],
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )
        slopes1 = torch.sum(torch.square(gradients[0]), dim=(1, 2))
        slopes2 = torch.sum(torch.square(gradients[1]), dim=(1))
        slopes = torch.sqrt(slopes1 + slopes2 + self.EPS)
        dis_loss_gp = torch.mean((slopes - 1.0) ** 2)

        return dis_loss_gp

    def _calculate_gp_attr_dis(
            self, batch_size, fake_attribute, real_attribute):
        alpha_dim2 = torch.FloatTensor(
            batch_size, 1).uniform_(1).to(
            self.device)
        differences_input_attribute = fake_attribute - real_attribute
        interpolates_input_attribute = real_attribute + (
            alpha_dim2 * differences_input_attribute
        )
        mixed_scores = self.attr_discriminator(interpolates_input_attribute)
        gradients = torch.autograd.grad(
            inputs=interpolates_input_attribute,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )
        slopes1 = torch.sum(torch.square(gradients[0]), dim=(1))

        slopes = torch.sqrt(slopes1 + self.EPS)
        attr_dis_gp = torch.mean((slopes - 1.0) ** 2)

        return attr_dis_gp

    def _train(self, dataset):
        if self.restore and self.pretrain_dir is None:
            raise ValueError("restore=True but no pretrain_dir is set!")
        if self.restore:
            if self.pretrain_dir is None:
                raise ValueError("restore=True but no pretrain_dir is set!")
            if not os.path.exists(self.pretrain_dir):
                raise ValueError(
                    f"Model path {self.pretrain_dir} does not exist!")
            print(f"Load pre-trained model from {self.pretrain_dir}...")
            self.load(self.pretrain_dir)

        self.generator.train()
        self.discriminator.train()
        if self.use_attr_discriminator:
            self.attr_discriminator.train()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size * self.num_packing,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2 + self.num_packing,
            persistent_workers=True,
        )
        iteration = 0
        loss_dict = {
            "g_loss_d": 0.0,
            "g_loss": 0.0,
            "d_loss_fake": 0.0,
            "d_loss_real": 0.0,
            "d_loss_gp": 0.0,
            "d_loss": 0.0,
        }
        if self.use_attr_discriminator:
            loss_dict["g_loss_attr_d"] = 0.0
            loss_dict["attr_d_loss_fake"] = 0.0
            loss_dict["attr_d_loss_real"] = 0.0
            loss_dict["attr_d_loss_gp"] = 0.0
            loss_dict["attr_d_loss"] = 0.0

        for epoch in tqdm(range(self.epochs)):
            with open(self.time_path, "a") as f:
                time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write("epoch {} starts: {}\n".format(epoch, time))

            for batch_idx, (real_attribute, real_feature) in enumerate(loader):

                real_attribute = real_attribute.to(self.device)
                real_feature = real_feature.to(self.device)
                real_attribute_noise = self._gen_attribute_input_noise(
                    self.batch_size
                ).to(self.device)
                addi_attribute_noise = self._gen_attribute_input_noise(
                    self.batch_size
                ).to(self.device)
                feature_input_noise = self._gen_feature_input_noise(
                    self.batch_size, self.sample_time
                ).to(self.device)

                for _ in range(self.d_rounds):
                    fake_attribute_list = []
                    fake_feature_list = []

                    for _ in range(self.num_packing):

                        h0 = Variable(
                            torch.normal(
                                0, 1, (self.generator.feature_num_layers,
                                       self.batch_size, self.generator.feature_num_units)
                            )).to(self.device)
                        c0 = Variable(
                            torch.normal(
                                0, 1, (self.generator.feature_num_layers,
                                       self.batch_size, self.generator.feature_num_units)
                            )).to(self.device)

                        fake_attribute, _, fake_feature = self.generator(
                            real_attribute_noise,
                            addi_attribute_noise,
                            feature_input_noise,
                            h0,
                            c0
                        )

                        fake_attribute_list.append(fake_attribute)
                        fake_feature_list.append(fake_feature)

                    packed_fake_attribute = torch.cat(
                        fake_attribute_list, dim=1)
                    packed_fake_feature = torch.cat(fake_feature_list, dim=1)

                    num_attrs = real_attribute.size()[1]
                    packed_real_attribute = real_attribute.view(
                        -1, self.num_packing * num_attrs
                    )

                    num_time_steps = real_feature.size()[1]
                    num_feats = real_feature.size()[2]
                    packed_real_feature = real_feature.view(
                        -1, self.num_packing * num_time_steps, num_feats
                    )

                    dis_real = self.discriminator(
                        packed_real_feature, packed_real_attribute
                    )
                    dis_fake = self.discriminator(
                        packed_fake_feature, packed_fake_attribute
                    )

                    dis_loss_fake = torch.mean(dis_fake)
                    dis_loss_real = -torch.mean(dis_real)

                    # dis_loss_fake_unflattened = dis_fake
                    # dis_loss_real_unflattened = -dis_real

                    dis_loss_gp = self._calculate_gp_dis(
                        batch_size=self.batch_size,
                        fake_feature=packed_fake_feature,
                        real_feature=packed_real_feature,
                        fake_attribute=packed_fake_attribute,
                        real_attribute=packed_real_attribute,
                    )
                    dis_loss_gp_term = self.d_gp_coe * dis_loss_gp

                    dis_loss = dis_loss_fake + dis_loss_real + dis_loss_gp_term
                    # dis_loss_unflattend = dis_loss_fake_unflattened + \
                    #     dis_loss_real_unflattened + self.d_gp_coe * dis_loss_gp_unflattend

                    self.opt_discriminator.zero_grad(set_to_none=False)
                    dis_loss.backward(retain_graph=True)
                    self.opt_discriminator.step()

                    loss_dict["d_loss_fake"] = dis_loss_fake
                    loss_dict["d_loss_real"] = dis_loss_real
                    loss_dict["d_loss_gp"] = dis_loss_gp
                    loss_dict["d_loss"] = dis_loss

                    if self.use_attr_discriminator:
                        attr_dis_real = self.attr_discriminator(
                            packed_real_attribute)
                        attr_dis_fake = self.attr_discriminator(
                            packed_fake_attribute)

                        attr_dis_loss_fake = torch.mean(attr_dis_fake)
                        attr_dis_loss_real = -torch.mean(attr_dis_real)

                        attr_dis_loss_fake_unflattened = attr_dis_fake
                        attr_dis_loss_real_unflattened = -attr_dis_real

                        attr_dis_loss_gp = self._calculate_gp_attr_dis(
                            batch_size=self.batch_size,
                            fake_attribute=packed_fake_attribute,
                            real_attribute=packed_real_attribute,
                        )
                        attr_dis_loss_gp_term = self.attr_d_gp_coe * attr_dis_loss_gp

                        attr_dis_loss = (
                            attr_dis_loss_fake
                            + attr_dis_loss_real
                            + attr_dis_loss_gp_term
                        )
                        # attr_diss_loss_unflattened = attr_dis_loss_fake_unflattened + \
                        #     attr_dis_loss_real_unflattened + self.attr_d_gp_coe * attr_dis_loss_gp_unflattened
                        self.opt_attr_discriminator.zero_grad(set_to_none=False)
                        attr_dis_loss.backward(retain_graph=True)
                        self.opt_attr_discriminator.step()

                        loss_dict["attr_d_loss_fake"] = attr_dis_loss_fake
                        loss_dict["attr_d_loss_real"] = attr_dis_loss_real
                        loss_dict["attr_d_loss_gp"] = attr_dis_loss_gp
                        loss_dict["attr_d_loss"] = attr_dis_loss

                for _ in range(self.g_rounds):
                    dis_fake = self.discriminator(
                        packed_fake_feature, packed_fake_attribute
                    )
                    gen_loss_dis = -torch.mean(dis_fake)
                    if self.use_attr_discriminator:
                        attr_dis_fake = self.attr_discriminator(
                            packed_fake_attribute)
                        gen_loss_attr_dis = -torch.mean(attr_dis_fake)
                        gen_loss = gen_loss_dis + self.g_attr_d_coe * gen_loss_attr_dis
                    else:
                        gen_loss = gen_loss_dis

                    self.opt_generator.zero_grad(set_to_none=False)
                    gen_loss.backward()
                    self.opt_generator.step()

                    loss_dict["g_loss_d"] = gen_loss_dis
                    if self.use_attr_discriminator:
                        loss_dict["g_loss_attr_d"] = gen_loss_attr_dis
                    loss_dict["g_loss"] = gen_loss

                self._write_losses(loss_dict, iteration)

                iteration += 1

            if (epoch + 1) % self.epoch_checkpoint_freq == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"epoch_id-{epoch}.pt")
                self.save(ckpt_path)

            with open(self.time_path, "a") as f:
                time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write("epoch {} ends: {}\n".format(epoch, time))

    def _generate(
        self,
        real_attribute_noise,
        addi_attribute_noise,
        feature_input_noise,
        h0,
        c0,
        given_attribute=None,
        given_attribute_discrete=None,
    ):

        self.generator.eval()
        self.discriminator.eval()
        if self.use_attr_discriminator:
            self.attr_discriminator.eval()

        if given_attribute is None and given_attribute_discrete is None:
            with torch.no_grad():
                attribute, attribute_discrete, feature = self.generator(
                    real_attribute_noise=real_attribute_noise.to(self.device),
                    addi_attribute_noise=addi_attribute_noise.to(self.device),
                    feature_input_noise=feature_input_noise.to(self.device),
                    h0=h0.to(self.device),
                    c0=c0.to(self.device)
                )
        else:
            given_attribute = torch.from_numpy(given_attribute).float()
            given_attribute_discrete = torch.from_numpy(
                given_attribute_discrete).float()
            with torch.no_grad():
                attribute, attribute_discrete, feature = self.generator(
                    real_attribute_noise=real_attribute_noise.to(self.device),
                    addi_attribute_noise=addi_attribute_noise.to(self.device),
                    feature_input_noise=feature_input_noise.to(self.device),
                    h0=h0.to(self.device),
                    c0=c0.to(self.device),
                    given_attribute=given_attribute.to(self.device),
                    given_attribute_discrete=given_attribute_discrete.to(
                        self.device))
        return attribute.cpu(), attribute_discrete.cpu(), feature.cpu()

    def _write_losses(self, loss_dict, iteration):

        self.writer.add_scalar("loss/g/d", loss_dict["g_loss_d"], iteration)
        if self.use_attr_discriminator:
            self.writer.add_scalar(
                "loss/g/attr_d", loss_dict["g_loss_attr_d"], iteration
            )
        self.writer.add_scalar("loss/g", loss_dict["g_loss"], iteration)

        self.writer.add_scalar(
            "loss/d/fake", loss_dict["d_loss_fake"],
            iteration)
        self.writer.add_scalar(
            "loss/d/real", loss_dict["d_loss_real"],
            iteration)
        self.writer.add_scalar("loss/d/gp", loss_dict["d_loss_gp"], iteration)
        self.writer.add_scalar("loss/d", loss_dict["d_loss"], iteration)

        if self.use_attr_discriminator:
            self.writer.add_scalar(
                "loss/attr_d/fake", loss_dict["attr_d_loss_fake"], iteration
            )
            self.writer.add_scalar(
                "loss/attr_d/real", loss_dict["attr_d_loss_real"], iteration
            )
            self.writer.add_scalar(
                "loss/attr_d/gp", loss_dict["attr_d_loss_gp"], iteration
            )
            self.writer.add_scalar(
                "loss/attr_d", loss_dict["attr_d_loss"],
                iteration)
