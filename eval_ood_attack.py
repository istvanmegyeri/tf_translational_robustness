from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from datareader import DataSet
from sklearn.metrics import roc_curve, average_precision_score, precision_recall_curve, auc
from collections import OrderedDict
import matplotlib.pyplot as plt
from attacks import Attack
import util
import pandas as pd
import os


def avg_auroc(y, p, disp=False):
    aucs = []
    for i in range(y.shape[1]):
        fpr, tpr, thresholds = roc_curve(y[:, i], p[:, i])
        if disp:
            plt.plot(fpr, tpr)
            plt.show()
        v = auc(fpr, tpr)
        aucs.append(v)
    return np.nanmean(aucs)


def avg_auprc(y, p, disp=False):
    aucs = []
    for i in range(y.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y[:, i], p[:, i])
        if disp:
            plt.plot(precision, recall)
            plt.show()
        v = auc(recall, precision)
        aucs.append(v)
    return np.nanmean(aucs)


def max_score(p):
    return np.max(p, axis=1, keepdims=True)


def mean_score(p):
    return np.mean(p, axis=1, keepdims=True)


def loss_score(p):
    return -np.sum(np.log(1 - p), axis=1, keepdims=True)


def avg_ps(y, p):
    return average_precision_score(y, p)


def make_attack(class_name, model, args) -> Attack:
    attack_cls = util.load_class(class_name)
    return attack_cls(model, **vars(args))


def main(params):
    print(params)
    score_fns = OrderedDict([('max', max_score),
                             ('mean', mean_score),
                             ('loss', loss_score)
                             ])
    ds = DataSet(params.data_path, params.test_path)
    model = tf.keras.models.load_model(params.m_path)
    x_test, y_test = ds.get_test()
    if params.head is not None:
        x_test = np.concatenate((x_test[:params.head], x_test[x_test.shape[0] // 2:params.head]), axis=0)
        y_test = np.concatenate((y_test[:params.head], y_test[y_test.shape[0] // 2:params.head]), axis=0)

    # generate ood samples
    attack = make_attack(params.attack, model, params)
    use_avg = True
    if use_avg:
        x_test_out, y_test_out = attack(x_test[:x_test.shape[0] // 2], y_test[:x_test.shape[0] // 2])
    else:
        x_test_out, y_test_out = attack(x_test, y_test)

    if use_avg:
        x_test_out = np.concatenate((x_test_out, np.flip(x_test_out, axis=1)), axis=0)
    x_all = np.concatenate((x_test, x_test_out), axis=0)
    del x_test, x_test_out

    # make prediction on normal and ood samples
    pred = model.predict(x_all, batch_size=500)
    if use_avg:
        y_test = y_test[:y_test.shape[0] // 2]
    pred_in = pred[:pred.shape[0] // 2]
    pred_out = pred[pred.shape[0] // 2:]
    if use_avg:
        pred_in = (pred_in[:pred_in.shape[0] // 2] + pred_in[pred_in.shape[0] // 2:]) / 2
        pred_out = (pred_out[:pred_out.shape[0] // 2] + pred_out[pred_out.shape[0] // 2:]) / 2

    base_stat = vars(params)

    # eval performance on the test set
    stats = [{**base_stat, 'set': 'in', 'avg-auroc': avg_auroc(y_test, pred_in),
              'avg-auprc': avg_auprc(y_test, pred_in)}]
    print(stats[-1])
    pred = np.concatenate((pred_in, pred_out), axis=0)
    is_in = np.zeros((pred.shape[0], 1))
    is_in[:is_in.shape[0] // 2] = 1

    # eval performance on in vs ood task
    for name, sf in score_fns.items():
        score = sf(pred)
        stats.append({**base_stat, 'set': attack.get_name(),
                      'score': name,
                      'avg-auroc': avg_auroc(is_in, score),
                      'avg-auprc': avg_auprc(is_in, score)
                      })
        print(stats[-1])
    
    if params.out_fname is not None:
        df = pd.DataFrame(data=stats)
        out_csv_name = params.out_fname
        util.mk_parent_dir(out_csv_name)
        df.to_csv(out_csv_name, index=False, mode='a', header=not os.path.exists(out_csv_name))


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--head', type=int)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--out_fname', type=str)
    parser.add_argument('--attack', type=str, required=True)
    parser.add_argument('--m_path', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=1)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        selected = gpus[FLAGS.gpu]
        tf.config.experimental.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.experimental.set_virtual_device_configuration(
            selected,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        with tf.device(l_gpu.name):
            main(FLAGS)
    else:
        main(FLAGS)
