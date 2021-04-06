from abc import ABC
from abc import abstractmethod
import tensorflow as tf
import numpy as np


class Attack(ABC):
    def __init__(self, model, **kwargs) -> None:
        self.model = model

    @abstractmethod
    def __call__(self, x, y):
        pass

    @abstractmethod
    def get_name(self):
        pass


class Shuffle(Attack):

    def __init__(self, model, seed=9, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seed = seed

    def get_name(self):
        return "Shuffle"

    def __call__(self, x, y):
        rnd = np.random.RandomState(self.seed)
        x_shuffle = np.array(x, copy=True)
        seq_length = x.shape[1]
        for i in range(x.shape[0]):
            idxs = rnd.permutation(seq_length)
            x_shuffle[i] = x[i][idxs]
        return x_shuffle, y


class Uniform(Attack):

    def __init__(self, model, seed=9, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seed = seed

    def get_name(self):
        return "Uniform"

    def __call__(self, x, y):
        rnd = np.random.RandomState(self.seed)
        x_uni = np.array(x, copy=True, dtype=np.float32)
        shape = x[0].shape
        for i in range(x_uni.shape[0]):
            x_uni[i] = rnd.uniform(0, 1, shape)
        return x_uni, y


class GenSeq(Attack):

    def __init__(self, model, seed=9, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seed = seed

    def get_name(self):
        return "GenSeq"

    def __call__(self, x, y):
        rnd = np.random.RandomState(self.seed)
        x_gen = np.zeros_like(x, dtype=np.float32)
        sample_shape = x[0].shape
        seq_idxs = np.arange(0, sample_shape[0])
        for i in range(x_gen.shape[0]):
            random_ids = rnd.randint(0, 4, sample_shape[0])
            x_gen[i, seq_idxs, random_ids] = 1
            # print(np.mean(x_gen[i], axis=0))
        return x_gen, y


class MiddleCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length

    def get_name(self):
        return "MiddleCrop"

    def __call__(self, x, y):
        dim_to_crop = x.shape[2] - self.seq_length
        return x[:, :, dim_to_crop // 2:-dim_to_crop // 2], y


class RandomCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length

    def get_name(self):
        return "RandomCrop"

    def __call__(self, x, y):
        x_crop = tf.image.random_crop(x, size=[x.shape[0], x.shape[1], self.seq_length, x.shape[3]])
        return x_crop, y


class WorstCrop(Attack):

    def __init__(self, model, seq_length, loss='zero-one', batch_size=64, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length
        self.batch_size = batch_size
        if loss == 'zero-one':
            self.loss = lambda p, y: np.argmax(p, axis=1) != np.argmax(y, axis=1)
        elif loss == 'mse':
            self.loss = lambda p, y: np.sum(np.square(p - y), axis=1)
        elif loss == 'xe':
            # Numerically unstable
            # self.loss = lambda p, y: -np.sum(np.log(p) * y, axis=1)
            # self.loss = lambda p, y: -np.log(np.sum(p * y, axis=1))
            eps = 1e-4
            self.loss = lambda p, y: -np.log(np.maximum(np.sum(p * y, axis=1), eps))

    def get_name(self):
        return "WorstCrop"

    def __call__(self, x, y):
        dim_to_crop = x.shape[2] - self.seq_length
        x_adv = x[:, :, :-dim_to_crop]
        pred = self.model.predict(x_adv, batch_size=self.batch_size)
        l_val = self.loss(pred, y)
        for i in range(1, dim_to_crop + 1):
            x_adv_i = x[:, :, i:x.shape[2] - (dim_to_crop - i)]
            pred_i = self.model.predict(x_adv_i, batch_size=self.batch_size)
            l_vali = self.loss(pred_i, y)
            improved = l_vali > l_val
            if np.any(improved):
                x_adv = tf.where(tf.reshape(improved, (improved.shape[0], 1, 1, 1)), x_adv_i, x_adv)
                l_val[improved] = l_vali[improved]
        return x_adv, y
