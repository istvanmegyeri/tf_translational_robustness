from abc import ABC
from abc import abstractmethod
from datareader import random_crop
import tensorflow as tf
import numpy as np


class Attack(ABC):
    def __init__(self, model, **kwargs) -> None:
        self.model = model

    @abstractmethod
    def __call__(self, x, y):
        pass


class MiddleCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length

    def __call__(self, x, y):
        dim_to_crop = x.shape[2] - self.seq_length
        return x[:, :, dim_to_crop // 2:-dim_to_crop // 2], y


class RandomCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length

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
        # @TODO: add support for categorical crossentropy

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
