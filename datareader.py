import numpy as np
from sklearn.model_selection import train_test_split


class DataSet():
    def __init__(self, f_path, test_path=None,
                 val_size=0.1,
                 seed=9) -> None:
        data = np.load(f_path)
        if 'x' in data.keys():
            x_train, y_train = data['x'], data['y']
            if test_path is not None:
                test_data = np.load(test_path)
                self.x_test, self.y_test = test_data['x'], test_data['y']
        else:
            x_train, y_train, self.x_test, self.y_test = data['arr_0'], data['arr_1'], data['arr_2'], \
                                                         data['arr_3']

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, test_size=val_size,
                                                                              random_state=seed)

    def get_train(self):
        return self.x_train, self.y_train

    def get_val(self):
        return self.x_val, self.y_val

    def get_test(self):
        return self.x_test, self.y_test
