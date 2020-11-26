import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class DataSet:
    def __init__(self, f_path, val_size=0.1,
                 seed=9) -> None:
        tf_data = np.load(f_path)
        x_train, y_train, self.x_test, self.y_test = tf_data['arr_0'], tf_data['arr_1'], tf_data['arr_2'], \
                                                     tf_data['arr_3']

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, test_size=val_size,
                                                                              random_state=seed)

    def get_train(self):
        return self.x_train, self.y_train

    def get_val(self):
        return self.x_val, self.y_val

    def get_test(self):
        return self.x_test, self.y_test
