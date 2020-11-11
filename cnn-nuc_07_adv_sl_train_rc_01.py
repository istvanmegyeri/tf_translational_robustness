import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# #define visible gpu device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Time and exp. data (e.g. TFs)
print(datetime.datetime.now())
start_time = datetime.datetime.now()

# adat_mappa = flank_MAFK_MA0496-1_NBS_NoSh+FlSh
# data\motif_discovery\wgEncodeAwgTfbsSydhK562Znf143IggrabUniPk

exp_info = "nuc_adv"
exp_desc = "seq_len_rc"  # random crop
run_num_for = 1

# run util. params
time_of_run = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
test_converted = 0
first_to_csv = 0

adat_mappa = "data/motif_discovery/wgEncodeAwgTfbsSydhImr90MafkIggrabUniPk/"
""" TF list
Mafk:   wgEncodeAwgTfbsSydhImr90MafkIggrabUniPk
Znf:    wgEncodeAwgTfbsSydhK562Znf143IggrabUniPk
"""
newpath = r"kiserlet/" + adat_mappa[36:] + "/" + exp_info + "/" + time_of_run  # exp name middle - fs: feature select

if not os.path.exists(newpath):
    os.makedirs(newpath)


def plot_history(history, gs=0, se=0, optimized_param=0):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('loss and val_loss')
    plt.plot(hist['epoch'], hist['loss'], label='loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='val_loss')
    plt.legend()
    plt.ylim([(min(hist['loss'].min(), hist['val_loss'].min())) * 0.9,
              (max(hist['loss'].max(), hist['val_loss'].max())) * 1.1])
    plt.savefig(newpath + r'/loss_val_loss_%d_%d_%s.png' % (gs, se, optimized_param))

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label='Validation Accuracy')
    plt.legend()
    plt.ylim([(min(hist['accuracy'].min(), hist['val_accuracy'].min())) * 0.9,
              (max(hist['accuracy'].max(), hist['val_accuracy'].max())) * 1.1])
    plt.savefig(newpath + r'/accuracy_val_accuracy_%d_%d_%s.png' % (gs, se, optimized_param))


# adatok betöltése
tf_data = np.load("data/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz")
x_train, y_train, x_test, y_test = tf_data['arr_0'], tf_data['arr_1'], tf_data['arr_2'], tf_data['arr_3']

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

# grid search:
gs, se = 0, 0

# # LR #optimized_param = 0.01 # LR
# nf = 1 # neuron number scale factor
# # for lr_gs in range(len(lr_gs_list)):
# #     for gs in range(len(optimized_param_list)):  # len(optimized_param_list
# #         for gs2 in range(len(optimized_param_list2)):

x_train_bckp, x_test_bckp = x_train.copy(), x_test.copy()

# data_augmentation = tf.keras.Sequential([
#   layers.experimental.preprocessing.PreprocessingLayer.Ran,
# ])
#    seq = tf.image.random_crop(x_train[0, :, :, :],
                               #size=[x_train[0, :, :, :].shape[0], x_train[0, :, :, :].shape[1] - 11, 4])
#

# 90-es véletlen vágásos augmentáció

def random_crop_90(seq, label):
    seq = tf.image.random_crop(seq,
                               size=[seq.shape[0], seq.shape[1] - 11, 4])
    return seq, label

# TF DataSet elkészítése

x_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
x_train = x_train.map(random_crop_90)  # ellenőrzés egy elem kikérésével: next(iter(x_train))
x_train = x_train.shuffle(buffer_size=len(x_train_bckp)).batch(64)

batch_size = 64

#x_train = (x_train.shuffle(1000).map(random_crop_90).batch(batch_size))

x_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
x_val = x_val.map(random_crop_90)
x_val = x_val.batch(64)

x_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
x_test = x_test.map(random_crop_90)
x_test = x_test.batch(64)


optimized_param = ""
optimized_param2 = ""
#for gs in range(6):
    # #optimized_param = str(optimized_param_list[gs])
    # seq_len_list = [90, 80, 70, 60, 50]
    # seq_len_list = [0, 10, 20, 30, 40, 50]
    # seq_len_list = [0, 5, 10, 15, 20, 25, ]
    # optimized_param = seq_len_list[gs]  # LR: "+str(lr_gs_list[lr_gs])
    # optimized_param2 = ""  # "L1: "+str(optimized_param_list[gs]) +"_" +"L2: "+ str(optimized_param_list2[gs2])
    # "ellenorzo fgdr-06" #"512,256}p" # használaton kívülre helyezve

    # Szekvencia hosszának csökkentése
    # x_train = x_train_bckp[:, :, optimized_param:100 - optimized_param, :]
    # x_test = x_test_bckp[:, :, optimized_param:100 - optimized_param, :]

model = tf.keras.models.Sequential([
    layers.Conv2D(256, (1, 24), input_shape=(tuple(x_train.element_spec[0].shape[1:])),
                  activity_regularizer=tf.keras.regularizers.l1(5e-5),
                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                  padding='same', ),
    #               padding='same', activation='relu', data_format="channels_first"), # channels_first , channels_last
    #               #activity_regularizer=regularizers.l1(0.01)),
    # kernel_regularizer=tf.keras.regularizers.l1(0.0001),
    layers.ReLU(),
    # layers.Conv2D(32 *nf, (1, 24), input_shape=(G_np_r_format_orig_test_norm_reshape.shape[1:]),
    # channels_first , channels_last
    # layers.Conv2D(16 *nf, (1, 12), padding='same', activation='relu', data_format="channels_first"),
    layers.Conv2D(64, (1, 12), padding='same',
                  activity_regularizer=tf.keras.regularizers.l1(5e-5),
                  kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    layers.ReLU(),

    # layers.Conv2D(8 *nf, (1, 6), padding='same', activation='relu', data_format="channels_first"),
    layers.GlobalMaxPooling2D(),  # O: pool_size=(2, 2)
    # layers.Dropout(0.2),
    # layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.5),
    # layers.Dense(150, activation='relu'),
    # layers.Dropout(0.1),
    layers.Dense(2, activation='softmax')

])

optimizer = tf.keras.optimizers.Adam(lr=5e-5)  # tf.keras.optimizers.Adam(lr=0.00001) #'adadelta'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                                  restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0, )

summary_out = model.summary()

epochs = 100

history = model.fit(x_train, epochs=epochs, verbose=2, validation_data=x_val,
                    callbacks=[early_stopping, reduce_lr])

model.save("models/nuc_adv_eval" + adat_mappa[36:-1] + "_" + time_of_run + str(gs) + "_" +
           exp_desc + ".hdf5")

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plot_history(history, gs, se, optimized_param=str(optimized_param) + str(optimized_param2))

print("Tanulás vége: ")
print(datetime.datetime.now())

# jupyter nbconvert --to script fgdr-05_02_nbconvert.ipynb
# test_converted = 0

struct_network = "m(256,64]ks(24,12)) - FGDR LR: 5e-5 l1: 5e-5 l2: 5e-4"
s2 = 0
# pool(24,12,6,3) FGDR LR01+reg l2 "
# "FGDR LR02+reg l2 m(c:256;256;(24)p2;256;256;(12)p4;" \
#                  "128;64(6)p8;64;32;(3)Mp;500;100)"

changes = "nukleotid adversarial zengx2 modell"  # O : first promising run #m(c32,16,p,16,8)
param_test = "randomCrop" #x_train.shape[2]  # "L2: " + str(optimized_param_list[gs])
# "LR: "+str(optimized_param_list2[gs])+ " L1: " + str(optimized_param_list[gs2])

eval_score = model.evaluate(x_test, verbose=2)
print("Test accuracy: ", eval_score[1])

# note down difference

# Save result's data to .tsv

val_max = np.max(hist.values[:, 3])
val_max = format(val_max, '.6f')
# print("Grid search number %d of %d." % (gs + 1, len(optimized_param_list)))
print("Validation maximum at run %d: " % se, val_max)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
ti_to_np = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

newpath_for_tsv = newpath[:-19]

# Check for global data collector for quick reference
if os.path.isfile(newpath_for_tsv + '/pcd_fs_01.tsv'):
    to_t_old = np.loadtxt(newpath_for_tsv + '/pcd_fs_01.tsv', delimiter='\t', dtype=np.dtype('U100'), skiprows=1)
    if len(to_t_old.shape) == 1:
        run_number = int(to_t_old[0]) + 1
        to_t_old = np.reshape(to_t_old, (1, to_t_old.shape[0]))
    else:
        run_number = int(to_t_old[0, 0]) + 1
else:
    to_t_old = np.empty((0, 7), dtype=np.dtype('U100'))
    np.savetxt(newpath_for_tsv + '/pcd_fs_01.tsv', to_t_old, delimiter='\t', fmt='%s',
               header='Run Num.\tTest acc.\tVal. acc.\tTrain acc.\tDate\tComment\tOptimized',
               comments="")
    run_number = 1

# len(to_t_old.shape)

# len(to_t.shape)

to_t = np.array(
    (run_number, round(eval_score[1], 6), val_max, round(np.max(hist.values[:, 1]), 6), ti_to_np, changes,
     param_test))
to_t = np.reshape(to_t, (to_t.shape + (1,)))
to_t = to_t.T

to_t_2 = np.insert(to_t_old, 0, to_t, axis=0)

np.savetxt(newpath_for_tsv + '/pcd_fs_01.tsv', to_t_2, delimiter='\t', fmt='%s',
           header='Run Num.\tTest acc.\tVal. acc.\tTrain acc.\tDate\tComment\tOptimized',
           comments="")

d = {'test acc': [format(eval_score[1], '.6f')],
     'val acc': [val_max],
     'train acc': [np.max(hist.values[:, 1])],
     'run #': [gs],
     'iter': [se],
     # 'optimized' : [optimized_param],
     # 'optimized2' : [optimized_param2],
     'param test': [param_test],
     'epochs': [hist.values[-1, 5]],
     'changes': [changes],

     }

df = pd.DataFrame(data=d, columns=['test acc', 'val acc', 'train acc', 'iter', 'run #',
                                   'param test', 'epochs', 'changes', ])  # 'optimized','optimized2'])

if first_to_csv == 0:
    df.to_csv(newpath + '/test_my_csv.csv', mode='a', header=True, index=False)  # ./results_for_search/80p/
else:
    df.to_csv(newpath + '/test_my_csv.csv', mode='a', header=False, index=False)  # ./results_for_search/80p/
first_to_csv = 1

# Grid search parameter increment
# optimized_param = optimized_param[gs]

# save model structure 01
struct_network = np.array(struct_network, ndmin=2, dtype=object)

struct_network = np.array(([struct_network], [s2], [exp_desc]), ndmin=2, dtype=object)
np.savetxt(newpath + "/model_complexity.txt", struct_network, fmt='%s')

# Time running
end_time_of_script = datetime.datetime.now()
print("Start 'Time of run': ", time_of_run)
print("Script finished at: ", end_time_of_script)
print("Running time: ", end_time_of_script - start_time)
