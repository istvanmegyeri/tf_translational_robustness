import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader():
    def __init__(self, f_path, ds_set,
                 val_size=0.1,
                 seed=9) -> None:
        self.ds_set = ds_set
        self.f_path = f_path
        self.val_size = val_size
        self.seed = seed
        self.data = np.load(self.f_path)

    def get_data(self):
        is_deep_sea = 'x' in self.data.keys()
        if is_deep_sea:
            return self.data['x'], self.data['y']
        else:
            x_train, y_train, x_test, y_test = self.data['arr_0'], self.data['arr_1'], self.data['arr_2'], \
                                               self.data['arr_3']
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size,
                                                              random_state=self.seed)
            if self.ds_set == "train":
                return x_train, y_train
            elif self.ds_set == 'val':
                return x_val, y_val
            elif self.ds_set == 'test':
                return x_test, y_test
            else:
                raise Exception('Unknown set option: {0}'.format(self.ds_set))
