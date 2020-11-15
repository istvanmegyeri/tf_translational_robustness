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
        self.fn = lambda args: random_crop(args[0], args[1], seq_length)

    def __call__(self, x, y):
        return tf.map_fn(self.fn, (x, y))


class WorstCrop(Attack):

    def __init__(self, model, seq_length, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.seq_length = seq_length
        self.fn = lambda args: random_crop(args[0], args[1], seq_length)

    def __call__(self, x, y):
        dim_to_crop = x.shape[2] - self.seq_length
        labs = np.argmax(y, axis=1)
        x_adv = x[:, :, :-dim_to_crop]
        pred = self.model.predict(x_adv)
        succes = pred != labs
        for i in range(1, dim_to_crop + 1):
            x_adv_i = x[:, :, i:x.shape[2] - (dim_to_crop - i)]
            pred_i = self.model.predict(x_adv_i)
            succes_i = pred_i != labs
            improved = np.logical_and(np.logical_not(succes), succes_i)
            if np.any(improved):
                x_adv[improved] = x_adv_i[improved]

        return x_adv, y
