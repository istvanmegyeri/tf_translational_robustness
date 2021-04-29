import numpy as np
from sklearn.model_selection import train_test_split
import os


class DataLoader():
    def __init__(self, f_path, ds_set=None,
                 val_size=0.1,
                 seed=9) -> None:
        self.ds_set = ds_set
        self.f_path = f_path
        self.val_size = val_size
        self.seed = seed
        self.data = np.load(self.f_path)

    def is_deep_sea(self):
        return 'x' in self.data.keys()

    def get_data(self, ds_set=None):
        if self.is_deep_sea():
            return self.data['x'], self.data['y']
        else:
            if ds_set is None and self.ds_set is None:
                raise Exception('ds_set must be defined')
            x_train, y_train, x_test, y_test = self.data['arr_0'], self.data['arr_1'], self.data['arr_2'], \
                                               self.data['arr_3']
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size,
                                                              random_state=self.seed)
            selected = ds_set if ds_set is not None else self.ds_set
            if selected == "train":
                return x_train[:, 0, :, :], y_train
            elif selected == 'val':
                return x_val[:, 0, :, :], y_val
            elif selected == 'test':
                return x_test[:, 0, :, :], y_test
            else:
                raise Exception('Unknown set option: {0}'.format(self.ds_set))


class DeepSea():

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def get_train(self):
        ds = DataLoader(f_path=os.path.join(self.data_dir, 'train_e.npz'))
        return ds.get_data()

    def get_val(self):
        ds = DataLoader(f_path=os.path.join(self.data_dir, 'valid_e.npz'))
        return ds.get_data()

    def get_test(self):
        ds = DataLoader(f_path=os.path.join(self.data_dir, 'test_e.npz'))
        return ds.get_data()


class ZengData(DataLoader):
    def get_train(self):
        return self.get_data('train')

    def get_val(self):
        return self.get_data('val')

    def get_test(self):
        return self.get_data('test')
