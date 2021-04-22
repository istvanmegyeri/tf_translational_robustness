from tensorflow import keras


class TFModel():

    def __init__(self, activity_regularizer=keras.regularizers.l1(5e-5), kernel_regularizer=keras.regularizers.l2(5e-4),
                 dropout=0.5) -> None:
        self.activity_regularizer = activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout

    def get_name(self):
        return 'tf_model'

    def build_model(self, input_shape):
        layers = [keras.layers.Conv1D(256, 24, input_shape=input_shape,
                                      activity_regularizer=self.activity_regularizer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      padding='same'),
                  keras.layers.ReLU(),
                  keras.layers.Conv1D(64, 12, padding='same',
                                      activity_regularizer=self.activity_regularizer,
                                      kernel_regularizer=self.kernel_regularizer,
                                      ),
                  keras.layers.ReLU(),
                  keras.layers.GlobalMaxPooling1D(),
                  keras.layers.Dense(500, activation='relu')]
        if self.dropout > 0:
            layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(2, activation='softmax'))
        model = keras.models.Sequential(layers)
        return model
