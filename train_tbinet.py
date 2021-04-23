import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from datareader import DeepSea
from models import TBinet
from train_zeng import make_attack
import time
from tqdm import tqdm


def main(params):
    ds = DeepSea(data_dir=params.data_dir)
    x_train, y_train = ds.get_train()
    x_val, y_val = ds.get_val()
    print(x_train.shape, x_train.dtype, y_train.shape)
    add_shuffle = params.add_shuf == 'Y'
    model_holder = TBinet()
    model = model_holder.build_model((params.seq_length, 4))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    attack = make_attack(params.attack, model, params)
    attack_val = attack

    def generator():
        while True:
            idxs = np.random.permutation(x_train.shape[0])
            for i in range(x_train.shape[0]):
                i_mapped = idxs[i]
                if not add_shuffle:
                    yield (x_train[i_mapped], y_train[i_mapped])
                else:
                    if np.random.rand() < 0.5:
                        yield (x_train[i_mapped], y_train[i_mapped])
                    else:
                        yield (np.random.permutation(x_train[i_mapped]), np.zeros_like(y_train[i_mapped]))

    tf_dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32),
                                                (tf.TensorShape(x_train[0].shape), tf.TensorShape([None])))
    batch_size = params.batch_size * 2 if add_shuffle else params.batch_size
    tf_dataset = iter(tf_dataset.batch(batch_size).prefetch(10))

    if params.save_dir is not None:
        os.makedirs(params.save_dir, exist_ok=True)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(params.save_dir, "tbinet_{epoch:02d}.hdf5"), verbose=1,
            save_best_only=False)
        callbacks = [checkpointer, CSVLogger(os.path.join(params.save_dir, 'metrics.csv'))]
    else:
        callbacks = []
    epochs = params.epoch
    steps_per_epoch = x_train.shape[0] // params.batch_size
    for cb in callbacks:
        cb.set_model(model)
        cb.set_params(
            {'batch_size': batch_size, 'epochs': epochs, 'steps': steps_per_epoch,
             'samples': x_train.shape[0] * 2 if add_shuffle else x_train.shape[0],
             'verbose': 0,
             'do_validation': True,
             'metrics': ['loss', 'val_loss']})
        cb.on_train_begin()
    for i in range(epochs):
        t0 = time.time()
        for cb in callbacks:
            cb.on_epoch_begin(i)
        hist_i = []
        for j in tqdm(range(steps_per_epoch)):
            (x_batch, y_batch) = next(tf_dataset)
            x_batcha, y_batcha = attack(x_batch, y_batch)
            hist = model.train_on_batch(x_batcha, y_batcha)
            hist_i.append(hist)
        x_vala, y_vala = attack_val(x_val, y_val)
        vala_eval = model.evaluate(x_vala, y_vala, verbose=0)
        train_eval = np.mean(hist_i, axis=0)
        stats = {'loss': train_eval, 'val_loss': vala_eval}
        for cb in callbacks:
            cb.on_epoch_end(i, stats)
            if i == (epochs - 1) or model.stop_training:
                cb.on_train_end()
        if params.verbose > 0:
            print(i, stats, '{:.2} min'.format((time.time() - t0) / 60))
        del hist_i, hist, vala_eval, train_eval, stats
        if model.stop_training:
            break


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)

    # training params
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--save_dir', type=str)
    # attack params
    parser.add_argument('--attack', type=str, default='attacks.MiddleCrop')
    parser.add_argument('--seq_length', type=int, default=1000)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--attack_batch', type=float, default=20 * 100)
    parser.add_argument('--n_try', type=int, default=20)
    # ds params
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--add_shuf', type=str, default='N')

    FLAGS = parser.parse_args()
    np.random.seed(9)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main(FLAGS)
