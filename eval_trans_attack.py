from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from datareader import DataLoader
import os
import pandas as pd
import util
from attacks import Attack
import time
import metrics
from collections import OrderedDict


def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))


def main(params):
    print(params)
    ds = DataLoader(params.data_path, ds_set=params.set)
    x, y = ds.get_data()
    print(x.shape, y.shape)
    model = tf.keras.models.load_model(params.model_path)
    attack = make_attack(params.attack, model, params)
    preds_test = model.predict(attack(x, y)[0], batch_size=params.batch_size)

    d = OrderedDict([
        ("ds", params.data_path),
        ("tf", None if ds.is_deep_sea() else params.data_path.split(os.path.sep)[-2]),
        ("mode", None if ds.is_deep_sea() else params.data_path.split(os.path.sep)[-3]),
        ("attack", attack.get_name()),
        ("seq_length", params.seq_length),
        ("train_attack",
         params.model_path.split("/")[-2].split('_')[-1] if ds.is_deep_sea() else params.model_path.split("/")[-3]),
        ("train_mode",
         params.model_path.split("/")[-2].split('_')[-3] if ds.is_deep_sea() else params.model_path.split("/")[-5]),
        ("path", params.model_path),
        ("time", time.asctime(time.localtime(time.time())))
    ])
    for m in params.metric.split(','):
        if m == 'acc':
            corr_preds_test = (np.argmax(preds_test, axis=1) == np.argmax(y, axis=1))
            score = np.sum(corr_preds_test) / x.shape[0]
        elif m == 'auc':
            score = metrics.avg_auroc(y, preds_test)
        elif m == 'aupr':
            score = metrics.avg_auprc(y, preds_test)
        elif m == 'xe':
            score = np.mean(np.log(np.maximum(np.sum(y * preds_test, axis=1), 1e-4)))
        elif m == 'bce':
            eps = 1e-4
            score = np.mean(np.mean(np.where(y,
                                             -np.log(np.maximum(preds_test, eps)),
                                             -np.log(np.maximum(1 - preds_test, eps))),
                                    axis=1))
        else:
            raise Exception('Unknown metric: {0}'.format(m))
        d['adv_' + m] = score
        print('adv_' + m, score)

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
    # attack parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--attack', type=str, default='attacks.RandomCrop')
    parser.add_argument('--seq_length', type=int, required=True)
    parser.add_argument('--attack_batch', type=int, default=64)
    parser.add_argument('--loss', type=str, required=True)
    parser.add_argument('--n_try', type=int)
    # dataset paremeters
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--data_path', type=str, required=True)
    # loaded model parameters
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    # output params
    parser.add_argument('--out_fname', type=str)

    FLAGS = parser.parse_args()
    np.random.seed(9)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main(FLAGS)
