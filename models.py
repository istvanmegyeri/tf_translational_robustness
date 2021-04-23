from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, multiply, Permute, RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model


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


class TBinet():

    def __init__(self, add_dropout=True) -> None:
        self.add_dropout = add_dropout

    def build_model(self, input_size):
        sequence_input = Input(shape=input_size)

        # Convolutional Layer
        output = Conv1D(320, kernel_size=26, padding="valid", activation="relu")(sequence_input)
        output = MaxPooling1D(pool_size=13, strides=13)(output)
        if self.add_dropout:
            output = Dropout(0.2)(output)

        # Attention Layer
        attention = Dense(1)(output)
        attention = Permute((2, 1))(attention)
        attention = Activation('softmax')(attention)
        attention = Permute((2, 1))(attention)
        attention = Lambda(lambda x: K.mean(x, axis=2), name='attention', output_shape=(75,))(attention)
        attention = RepeatVector(320)(attention)
        attention = Permute((2, 1))(attention)
        output = multiply([output, attention])

        # BiLSTM Layer
        output = Bidirectional(LSTM(320, return_sequences=True, recurrent_activation="hard_sigmoid", implementation=1))(
            output)
        if self.add_dropout:
            output = Dropout(0.5)(output)

        flat_output = Flatten()(output)

        # FC Layer
        FC_output = Dense(695)(flat_output)
        FC_output = Activation('relu')(FC_output)

        # Output Layer
        output = Dense(690)(FC_output)
        output = Activation('sigmoid')(output)

        model = Model(inputs=sequence_input, outputs=output)
        return model
