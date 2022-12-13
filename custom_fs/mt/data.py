import gc
import tqdm
import numpy as np
import torch
from fs_mol.data import DataFold
from fs_mol.data import StratifiedTaskSampler
from fs_mol.data import DatasetTooSmallException


np.seterr(divide='ignore', invalid='ignore')


def get_label_matrix(labels, ids, dim):

    matrix = torch.zeros((len(labels), dim))

    for i in range(len(labels)):
        if not labels[i]:
            matrix[i][ids[i]] = -1
        else:
            matrix[i][ids[i]] = 1

    return (matrix + 1) / 2


class Normalizer:

    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.x_range = None

    def fit(self, x):
        self.x_min = np.min(x, axis=0)
        self.x_max = np.max(x, axis=0)
        self.x_range = self.x_max - self.x_min

    def transform(self, x):
        x_norm = (x - self.x_min) / self.x_range
        x_norm = np.nan_to_num(x_norm, copy=False)

        return x_norm


class FineTuningLoader:

    def __init__(self, dataset, DataFold, n_support, scaler, seed):
        self.seed = seed
        self.task_iterable = iter(dataset.get_task_reading_iterable(DataFold))
        self.sampler = StratifiedTaskSampler(n_support, 0.0, 256, allow_smaller_test=True)
        self.scaler = scaler

    def __iter__(self):
        return self

    def __next__(self):

        while True:
            try:
                task = next(self.task_iterable)
                sample = self.sampler.sample(task, seed=self.seed)
                break
            except DatasetTooSmallException:
                continue


        train_data = [[], []]
        test_data = [[], []]

        for train_sample in sample.train_samples:
            features = np.concatenate((train_sample.descriptors, train_sample.fingerprint)).astype(np.float32)
            train_data[0].append(features)
            train_data[1].append(np.array(train_sample.bool_label, dtype=np.float32))

        train_data[0] = self.scaler.transform(np.array(train_data[0]))

        for test_sample in sample.test_samples:
            features = np.concatenate((test_sample.descriptors, test_sample.fingerprint)).astype(np.float32)
            test_data[0].append(features)
            test_data[1].append(np.array(test_sample.bool_label, dtype=np.float32))

        test_data[0] = self.scaler.transform(np.array(test_data[0]))

        train_data = [(x, y) for x, y in zip(train_data[0], train_data[1])]
        test_data = [(x, y) for x, y in zip(test_data[0], test_data[1])]

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=False)

        pos_label_ratio = {"train": sample.train_pos_label_ratio,
                           "test": sample.test_pos_label_ratio}

        return train_dataloader, test_dataloader, pos_label_ratio, task.name


class MTDataset:

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.train_set = None
        self.finetuning_set = None
        self.scaler = Normalizer()

    def build_train_set(self):

        task_iterable = self.dataset.get_task_reading_iterable(DataFold.TRAIN)
        data = [[], [], []]

        for task_id, task in tqdm.tqdm(enumerate(task_iterable)):
            for sample in task.samples:
                features = np.concatenate((sample.descriptors, sample.fingerprint)).astype(np.float32)
                label = sample.bool_label
                data[0].append(features)
                data[1].append(label)
                data[2].append(task_id)

        gc.collect()

        x = np.array(data[0])
        self.scaler.fit(x)
        x_norm = self.scaler.transform(x)
        data[0] = x_norm

        self.train_set = [(x, y, z) for x, y, z in zip(data[0], data[1], data[2])]

    def build_finetuning_set(self, n_support, DataFold):

        self.finetuning_set = FineTuningLoader(self.dataset, DataFold, n_support, self.scaler, self.seed)

