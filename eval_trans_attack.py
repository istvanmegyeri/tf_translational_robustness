from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from datareader import DataSet
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
    ds = DataSet(params.data_path)
    if params.set == 'val':
        x, y = ds.get_val()
    elif params.set == "test":
        x, y = ds.get_test()
    else:
        raise Exception('Unsupported set: {0}'.format(params.set))
    model = tf.keras.models.load_model(params.model_path)
    attack = make_attack(params.attack, model, params)
    preds_test = model.predict(attack(x, y)[0])

    corr_preds_test = (np.argmax(preds_test, axis=1) == np.argmax(y, axis=1))
    test_acc = np.sum(corr_preds_test) / x.shape[0]

    d = {
        "ds": params.data_path,
        "adv_acc": test_acc,
        "attack": attack.get_name(),
        "train_mode": params.model_path.rsplit("/", 3)[1],
        "path": params.model_path,
        "time": time.asctime(time.localtime(time.time())),
    }
    print(d)
    if params.out_fname is not None:
        df = pd.DataFrame(data=[d])
        out_csv_name = params.out_fname
        util.mk_parent_dir(out_csv_name)
        df.to_csv(out_csv_name, index=False, mode='a', header=not os.path.exists(out_csv_name))


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--attack', type=str, default='attacks.RandomCrop')
    parser.add_argument('--loss', type=str, default='zero-one')
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--out_fname', type=str)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str,
                        default='./data/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz')
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
