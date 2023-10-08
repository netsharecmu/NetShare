from distutils.command.config import config
import os, math, copy
from more_itertools import sample
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from .util import add_gen_flag, normalize_per_sample

class NpyFolderDataset(object):
    def __init__(self, root, buffer_size=1000, num_processes=10):
        self.root = root
        self.buffer_size = buffer_size
        self.num_processes = num_processes

        self.files = [os.path.join(root, "data_train_npz", file) for file in os.listdir(os.path.join(root, "data_train_npz")) if file.endswith(".npz")]

        self.image_buffer = mp.Queue(self.buffer_size)
        self.running_flag = mp.Value("i", 1)
        self.processes = []
        for i in range(num_processes):
            process = mp.Process(
                target=self.data_loader,
                args=(self.running_flag,
                      self.image_buffer,
                      self.files,
                      self.transform))
            process.start()
            self.processes.append(process)

        # sample_data = self.sample_batch(1)
        # self.image_height = sample_data.shape[1]
        # self.image_width = sample_data.shape[2]
        # self.image_depth = sample_data.shape[3]

    @staticmethod
    def data_loader(running_flag, image_buffer, files, transform):
        np.random.seed(os.getpid())
        while running_flag.value:
            file_id = np.random.choice(len(files))
            image = np.load(files[file_id])
            image_ = {}
            for k in image.files:
                image_[k] = image[k]
            image_ = transform(image_)
            image_buffer.put(image_)
        image_buffer.cancel_join_thread()

    def stop_data_loader(self):
        self.running_flag.value = 0
        # while not self.image_buffer.empty():
        #     try:
        #         self.image_buffer.get_nowait()
        #     except mp.Queue.Empty:
        #         pass

        for process in self.processes:
            #process.join()
            process.terminate()

    def transform(self, image):
        return image

    def sample_batch(self, batch_size):
        images = []
        for i in range(batch_size):
            images.append(self.image_buffer.get())
        images = np.stack(images, axis=0)
        return images


class ILSVRC2012Dataset(NpyFolderDataset):
    def __init__(self, *args, **kwargs):
        super(ILSVRC2012Dataset, self).__init__(*args, **kwargs)

        if self.num_classes != 1000:
            raise Exception("Number of classes should be 1000")

        if self.num_images != 1281167:
            raise Exception("Number of images should be 1281167")

    @staticmethod
    def transform(images):
        images = images.astype(np.float32)
        images = images / 255.0 * 2.0 - 1.0
        return images

class NetShareDataset(NpyFolderDataset):
    def __init__(self, config, data_attribute_outputs, data_feature_outputs, *args, **kwargs):
        self.config = config
        self.data_attribute_outputs_orig = data_attribute_outputs # immutable
        self.data_feature_outputs_orig = data_feature_outputs # immutable
        self.data_attribute_outputs_train = None # mutable, feed in training
        self.data_feature_outputs_train = None # mutable, feed in training
        self.real_attribute_mask = None # mutable, feed in training
        self.gt_lengths = None # mutable, feed in training
        super(NetShareDataset, self).__init__(*args, **kwargs)

    # append to global_max_flow_len
    def transform(self, image):
        image_ = {}

        data_attribute = image["data_attribute"]
        data_feature = image["data_feature"]
        data_gen_flag = image["data_gen_flag"]

        # pad to multiple of sample_len
        max_flow_len = image["global_max_flow_len"][0]
        ceil_timeseries_len = math.ceil(max_flow_len / self.config["sample_len"]) * self.config["sample_len"]
        data_feature = np.pad(data_feature, pad_width=((0, ceil_timeseries_len - data_feature.shape[0]), (0, 0)), mode='constant', constant_values=0)
        data_gen_flag = np.pad(data_gen_flag, pad_width=(0, ceil_timeseries_len - data_gen_flag.shape[0]), mode='constant', constant_values=0)

        image_["data_attribute"] = data_attribute
        image_["data_feature"] = data_feature
        image_["data_gen_flag"] = data_gen_flag
        # image_["row_id"] = image["row_id"][0]

        # print(data_attribute.shape)
        # print(data_feature.shape)
        # print(data_gen_flag.shape)

        return image_
    
    def sample_batch(self, batch_size):
        data_attribute = []
        data_feature = []
        data_gen_flag = []
        for i in range(batch_size):
            image = self.image_buffer.get()
            data_attribute.append(image["data_attribute"])
            data_feature.append(image["data_feature"])
            data_gen_flag.append(image["data_gen_flag"])

        data_attribute = np.stack(data_attribute, axis=0)
        data_feature = np.stack(data_feature, axis=0)
        data_gen_flag = np.stack(data_gen_flag, axis=0)

        # print(data_attribute.shape)
        # print(data_feature.shape)
        # print(data_gen_flag.shape)

        if self.config["self_norm"]:
            (data_feature, data_attribute, self.data_attribute_outputs_train,
             self.real_attribute_mask) = \
                normalize_per_sample(
                    data_feature, data_attribute, data_gen_flag, 
                    self.data_feature_outputs_orig, self.data_attribute_outputs_orig)
        else:
            self.real_attribute_mask = [True] * len(self.data_attribute_outputs_orig)
            self.data_attribute_outputs_train = copy.deepcopy(self.data_attribute_outputs_orig)
        
        sample_len = self.config["sample_len"]
        if self.config["use_gt_lengths"]:
            self.data_feature_outputs_train = copy.deepcopy(self.data_feature_outputs_orig)
            self.gt_lengths = np.load(os.path.join(self.root, "gt_lengths.npy"))
        else:
            data_feature, self.data_feature_outputs_train = add_gen_flag(
                data_feature, data_gen_flag, self.data_feature_outputs_orig, sample_len)
            self.gt_lengths = None
        
        data_gen_flag = np.expand_dims(data_gen_flag, 2)
        
        return data_attribute, data_feature, data_gen_flag
    
    def sample_batch_with_rowid(self, batch_size):
        data_attribute = []
        data_feature = []
        data_gen_flag = []
        data_row_ids = []
        for i in range(batch_size):
            image = self.image_buffer.get()
            data_attribute.append(image["data_attribute"])
            data_feature.append(image["data_feature"])
            data_gen_flag.append(image["data_gen_flag"])
            data_row_ids.append(image["row_id"])

        data_attribute = np.stack(data_attribute, axis=0)
        data_feature = np.stack(data_feature, axis=0)
        data_gen_flag = np.stack(data_gen_flag, axis=0)

        # print(data_attribute.shape)
        # print(data_feature.shape)
        # print(data_gen_flag.shape)

        if self.config["self_norm"]:
            (data_feature, data_attribute, self.data_attribute_outputs_train,
             self.real_attribute_mask) = \
                normalize_per_sample(
                    data_feature, data_attribute, data_gen_flag, 
                    self.data_feature_outputs_orig, self.data_attribute_outputs_orig)
        else:
            self.real_attribute_mask = [True] * len(self.data_attribute_outputs_orig)
            self.data_attribute_outputs_train = copy.deepcopy(self.data_attribute_outputs_orig)
        
        sample_len = self.config["sample_len"]
        if self.config["use_gt_lengths"]:
            self.data_feature_outputs_train = copy.deepcopy(self.data_feature_outputs_orig)
            self.gt_lengths = np.load(os.path.join(self.root, "gt_lengths.npy"))
        else:
            data_feature, self.data_feature_outputs_train = add_gen_flag(
                data_feature, data_gen_flag, self.data_feature_outputs_orig, sample_len)
            self.gt_lengths = None
        
        data_gen_flag = np.expand_dims(data_gen_flag, 2)
        
        return data_attribute, data_feature, data_gen_flag, data_row_ids

# DO NOT USE THIS A TEST AS ``from .util import XX'' WILL CAUSE ERROR
# if __name__ == "__main__":
    

