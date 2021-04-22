from argparse import ArgumentParser
from tensorflow import keras
import tensorflow as tf
import numpy as np
from datareader import ZengData
from models import TFModel
import os
import pandas as pd
import util
from attacks import Attack
import time


def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))


def main(params):
    print(params)
    ds = ZengData(f_path=params.fname)
    x_train, y_train = ds.get_train()
    x_val, y_val = ds.get_val()
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    tf_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    tf_dataset = tf_dataset.shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True).batch(params.batch_size)
    if params.reg == "NO":
        model_holder = TFModel(activity_regularizer=None, kernel_regularizer=None, dropout=0)
    else:
        model_holder = TFModel()

    model = model_holder.build_model((params.seq_length, x_train.shape[2]))
    optimizer = keras.optimizers.Adam(lr=5e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    attack = make_attack(params.attack, model, params)

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                               mode='min',
                                               restore_best_weights=True, verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, verbose=1, mode='min',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
                 ]
    if params.save_dir is not None:
        m_path = os.path.join(params.save_dir, 'checkpoints', model_holder.get_name())
        util.mk_parent_dir(m_path)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(filepath=m_path + '_{epoch:03d}_{val_loss:.2f}_{val_accuracy:.2f}.hdf5'))
    epochs = params.epoch
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': params.batch_size, 'epochs': epochs, 'steps': x_train.shape[0] // params.batch_size,
             'samples': x_train.shape[0], 'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']})
        cb.on_train_begin()
    history = []
    for i in range(epochs):
        t0 = time.time()
        for cb in callbacks:
            cb.on_epoch_begin(i)
        hist_i = []
        for b_idx, (x_batch, y_batch) in enumerate(tf_dataset):
            x_batcha, y_batcha = attack(x_batch, y_batch)
            hist = model.train_on_batch(x_batcha, y_batcha)
            hist_i.append(hist)
        x_vala, y_vala = attack(x_val, y_val)
        vala_eval = model.evaluate(x_vala, y_vala, verbose=0)
        train_eval = np.mean(hist_i, axis=0)
        stats = {'loss': train_eval[0], 'accuracy': train_eval[1], 'val_loss': vala_eval[0],
                 'val_accuracy': vala_eval[1]}
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (epochs - 1) or model.stop_training:
                cb.on_train_end()
        history.append(
            {**stats, 'epoch': i, 'lr': model.optimizer.lr.numpy(), 'stop_training': model.stop_training})
        if params.verbose > 0:
            print(history[-1], '{:.2} min'.format((time.time() - t0) / 60))
        del hist_i, hist, vala_eval, train_eval, stats
        if model.stop_training:
            break
    if params.save_dir is not None:
        out_fname = os.path.join(params.save_dir, 'logs.csv')
        util.mk_parent_dir(out_fname)
        pd.DataFrame(history).to_csv(out_fname, index=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    # model params
    parser.add_argument('--reg', type=str, default='preset')

    # training params
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--save_dir', type=str)
    # attack params
    parser.add_argument('--attack', type=str, default='attacks.RandomCrop')
    parser.add_argument('--seq_length', type=int, default=90)
    parser.add_argument('--loss', type=str, default='xe')
    # ds params
    parser.add_argument('--fname', type=str, required=True)

    FLAGS = parser.parse_args()
    np.random.seed(9)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main(FLAGS)
