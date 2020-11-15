from tensorflow import keras


class TFModel():
    def get_name(self):
        return 'tf_model'

    def build_model(self, input_shape):
        model = keras.models.Sequential([
            keras.layers.Conv2D(256, (1, 24), input_shape=input_shape,
                                activity_regularizer=keras.regularizers.l1(5e-5),
                                kernel_regularizer=keras.regularizers.l2(0.0005),
                                padding='same'),
            keras.layers.ReLU(),
            keras.layers.Conv2D(64, (1, 12), padding='same',
                                activity_regularizer=keras.regularizers.l1(5e-5),
                                kernel_regularizer=keras.regularizers.l2(0.0005)),
            keras.layers.ReLU(),
            keras.layers.GlobalMaxPooling2D(),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')

        ])
        return model
