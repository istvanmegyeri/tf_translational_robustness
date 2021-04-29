import numpy as np
import h5py
import scipy.io
from sklearn import metrics
import pandas as pd
import os

# os.environ['THEANO_FLAGS'] = "device=cuda0,force_device=True,floatX=float32,gpuarray.preallocate=0.3"
# import theano
# print(theano.config.device)
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Layer, Input, Concatenate, Reshape, \
    concatenate, Lambda, multiply, Permute, Reshape, RepeatVector
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

# #define visible gpu device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ### Load data (training and validation)

# In[2]:


data_folder = "./data/"

# x_val = np.transpose(validmat['validxdata'], axes=(0, 2, 1))
# y_val = validmat['validdata'][:, 125:815]

# trainmat = h5py.File(data_folder + 'train.mat')
# validmat = scipy.io.loadmat(data_folder + 'valid.mat')
#
trainmat = np.load(data_folder+ 'train.npz')
x_train = trainmat['x']
# X_train = np.transpose(X_train, axes=(1, 0, 2))
y_train = trainmat['y']
# y_train = y_train[:, 125:815]
#
# trainmat.close()
#
# # ### Choose only the targets that correspond to the TF binding
# #
# X_train = x_val
# y_train = y_val


# X_train = x_val
# y_train = y_val
# y_train = y_train  # [:,125:815]

# y = np.zeros((100, 690, 1))
add_shuffle = True


# Adatgenerátor a keveréshez
def generate():
    while True:
        for i in range(X_train.shape[0]):
            if not add_shuffle:
                yield (X_train[i], y_train[i])
            else:
                if np.random.rand() < 0.5:
                    yield (X_train[i], y_train[i])
                else:
                    yield (np.random.permutation(X_train[i]), np.zeros_like(y_train[i]))
        idxs=np.random.permutation(X_train.shape[0])
        X_train=X_train[idxs]
        y_train=y_train[idxs]


ds = tf.data.Dataset.from_generator(generate, (tf.float32, tf.float32),
                                    (tf.TensorShape([1000, 4]), tf.TensorShape([None])))
# ds=tf.data.Dataset.from_generator(generate(X_train=X_train_test, y_train=y_test), (tf.float32, tf.float32),
#                                (tf.TensorShape([]), tf.TensorShape([None])))
batch_size = 200
# @ define shuffle parameters, e.g. buffer size
# ds=ds.shuffle().batch(batch_size)
ds = ds.batch(batch_size)

#
# # Adatgenerátor a keveréshez - validációs halmaz
# def generate_val():
#     while True:
#         for i in range(x_val.shape[0]):
#             if not add_shuffle:
#                 yield (x_val[i], y_val[i])
#             else:
#                 if np.random.rand() < 0.5:
#                     yield (x_val[i], y_val[i])
#                 else:
#                     yield (np.random.permutation(x_val[i]), np.zeros_like(y_val[i]))
#
#
# ds_val = tf.data.Dataset.from_generator(generate_val, (tf.float32, tf.float32),
#                                         (tf.TensorShape([1000, 4]), tf.TensorShape([None])))
# # ds=tf.data.Dataset.from_generator(generate(X_train=X_train_test, y_train=y_test), (tf.float32, tf.float32),
# #                                (tf.TensorShape([]), tf.TensorShape([None])))
# # @ define shuffle parameters, e.g. buffer size
# ds_val = ds_val.shuffle(buffer_size=len(x_val)).batch(batch_size)
# # X_train_train = X_train_train.shuffle(buffer_size=len(X_train_train_bckp)).batch(64)

"""
Modell Betöltése, Tanítás
"""

model = load_model("./model/tbinet_sajat_tf2_hs_02_d2.h5")

#
# # ### Run TBiNet
#
# sequence_input = Input(shape=(1000, 4))
#
# # Convolutional Layer
# output = Conv1D(320, kernel_size=26, padding="valid", activation="relu")(sequence_input)
# output = MaxPooling1D(pool_size=13, strides=13)(output)
# output = Dropout(0.2)(output)
#
# # Attention Layer
# attention = Dense(1)(output)
# attention = Permute((2, 1))(attention)
# attention = Activation('softmax')(attention)
# attention = Permute((2, 1))(attention)
# attention = Lambda(lambda x: K.mean(x, axis=2), name='attention', output_shape=(75,))(attention)
# attention = RepeatVector(320)(attention)
# attention = Permute((2, 1))(attention)
# output = multiply([output, attention])
#
# # BiLSTM Layer
# output = Bidirectional(LSTM(320, return_sequences=True, recurrent_activation="hard_sigmoid", implementation=1))(output)
# output = Dropout(0.5)(output)
#
# flat_output = Flatten()(output)
#
# # FC Layer
# FC_output = Dense(695)(flat_output)
# FC_output = Activation('relu')(FC_output)
#
# # Output Layer
# output = Dense(690)(FC_output)
# output = Activation('sigmoid')(output)
#
# model = Model(inputs=sequence_input, outputs=output)
#
# print('compiling model')
# model.compile(loss='binary_crossentropy', optimizer='adam')

print('model summary')
model.summary()

checkpointer = ModelCheckpoint(filepath="./model/tbinet_kevgen_05_.{epoch:02d}.hdf5", verbose=1,
                               save_best_only=False)
# earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

model.fit(ds, epochs=60, steps_per_epoch=2 * (x_train.shape[0] // batch_size), shuffle=True, verbose=1,
          # validation_data=ds_val, validation_steps=2 * (x_val.shape[0] // batch_size),
          callbacks=[checkpointer])
# model.fit(ds, epochs=1,steps_per_epoch=(X_train.shape[0]//batch_size), callbacks=[earlystopper, checkpointer])

model.save('./model/tbinet_sajat_kevgen_05_d1.h5')
