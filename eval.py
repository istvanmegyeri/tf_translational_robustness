from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from datareader import DeepSea
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, average_precision_score, precision_recall_curve, auc
from collections import OrderedDict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def avg_auc(y, p, disp=False):
    aucs = []
    # for i in range(y.shape[1]):
    for i in range(y.shape[1]):
        fpr, tpr, thresholds = roc_curve(y[:, i], p[:, i])
        if disp:
            plt.plot(fpr, tpr)
            plt.show()
        v = auc(fpr, tpr)
        aucs.append(v)
    return np.nanmean(aucs)


def avg_psauc(y, p, disp=False):
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


def ul_score(p):
    return np.mean(np.log(p), axis=1, keepdims=True)


def avg_ps(y, p):
    return average_precision_score(y, p)


def main(params):
    print(params)
    ds = DeepSea(params.fname)
    y_test = ds.get_test()
    pred = np.load('models/tbinet/tpreds_ossz_01.npz')['arr_0']
    pred_in = pred[:pred.shape[0] // 2]
    pred_out = pred[pred.shape[0] // 2:]
    idx = 0
    y_test = y_test[:y_test.shape[0] // 2]
    pred_in = (pred_in[:pred_in.shape[0] // 2] + pred_in[pred_in.shape[0] // 2:]) / 2
    pred_out = (pred_out[:pred_out.shape[0] // 2] + pred_out[pred_out.shape[0] // 2:]) / 2
    print('AVG-auc:', avg_auc(y_test, pred_in))
    print('AVG-PR-AUC:', avg_psauc(y_test, pred_in))
    score_fns = OrderedDict([('max', max_score),
                             ('mean', mean_score),
                             ('ul', ul_score)
                             ])
    print(np.sum(np.square(pred_in - pred_out)))
    print(pred_in.shape, pred_out.shape)
    pred = np.concatenate((pred_in, pred_out), axis=0)
    print(pred.shape)
    is_in = np.zeros((pred.shape[0], 1))
    is_in[:is_in.shape[0] // 2] = 1
    for name, sf in score_fns.items():
        score = sf(pred)
        print(name, 'AUC(in vs shuffle):', avg_auc(is_in, score))
        print(name, 'AUPR(in vs shuffle):', avg_psauc(is_in, score))
    is_in = is_in.flatten()
    pca = PCA(n_components=100)
    xs = pca.fit_transform(pred)
    for c in [0, 1]:
        plt.figure()
        plt.scatter(xs[is_in == c, 0], xs[is_in == c, 1])
    print((pca.explained_variance_ / np.sum(pca.explained_variance_))[:10])
    print(pca.explained_variance_ratio_[:10])
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--fname', type=str, required=True)
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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        with tf.device(l_gpu.name):
            main(FLAGS)
    else:
        main(FLAGS)
