from argparse import ArgumentParser
from tensorflow import keras
import tensorflow as tf
import numpy as np
from datareader import DataSet
from models import TFModel
import sys, os
import pandas as pd
import util
from attacks import Attack
import time


def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))


def main(params):
    print(params)
    ds = DataSet(params.data_path)
    x_test, y_test = ds.get_test()
    tf_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    tf_dataset = tf_dataset.shuffle(buffer_size=len(x_test), reshuffle_each_iteration=True).batch(params.batch_size)
    model = tf.keras.models.load_model(params.model_path)
    optimizer = keras.optimizers.Adam(lr=5e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    attack = make_attack(params.attack, model, params)
    preds_test = model.predict(attack(x_test, y_test)[0])

    corr_preds_test = (np.argmax(preds_test, axis=1) == np.argmax(y_test, axis=1))

    test_acc = np.sum(corr_preds_test)/x_test.shape[0]

    d = {
        "test acc": test_acc,
        "attack": attack.get_name(),
        "model": params.model_path.rsplit("/", 3)[1],
        "path": params.model_path,
        "time": time.asctime(time.localtime(time.time())),
    }

    df = pd.DataFrame(data=[d])
    # df = pd.DataFrame(data=d, columns=['test acc', 'attack', 'model', 'time'])
    if os.path.exists("models/test_eval.csv"):
        df.to_csv("models/test_eval.csv", index=False, mode='a', header=False)
    else:
        df.to_csv("models/test_eval.csv", index=False, mode='a', header=True)


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seq_length', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--attack', type=str, default='attacks.RandomCrop')
    # parser.add_argument('--attack', type=lambda x:x.split(','), default='attacks.RandomCrop,attacks.WorstCrop')
    parser.add_argument('--loss', type=str, default='zero-one')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--data_path', type=str, default='./data/motif_discovery/SydhImr90MafkIggrabUniPk'
                                                         '/SydhImr90MafkIggrabUniPk.npz')
    parser.add_argument('--model_path', type=str, default='./models/middle_/checkpoints/tf_model_038_0.27_0.94.hdf5')
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        selected = gpus[FLAGS.gpu]
        tf.config.experimental.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.experimental.set_virtual_device_configuration(
            selected,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        with tf.device(l_gpu.name):
            main(FLAGS)
    else:
        main(FLAGS)
