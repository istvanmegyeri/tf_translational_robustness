from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from datareader import DeepSea
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, average_precision_score, precision_recall_curve, auc


def avg_auc(y, p, disp=False):
    aucs = []
    # for i in range(y.shape[1]):
    for i in range(y.shape[1]):
        fpr, tpr, thresholds = roc_curve(y[:, i], p[:, i])
        if disp:
            plt.plot(fpr, tpr)
            plt.show()
        aucs.append(auc(fpr, tpr))
    return np.mean(aucs)


def avg_psauc(y, p, disp=False):
    aucs = []
    for i in range(y.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y[:, i], p[:, i])
        if disp:
            plt.plot(precision, recall)
            plt.show()
        aucs.append(auc(recall, precision))
    return np.mean(aucs)


def score_fn(p):
    return np.max(p, axis=1, keepdims=True)


def avg_ps(y, p):
    return average_precision_score(y, p)


def main(params):
    print(params)
    ds = DeepSea(params.fname)
    y_test = ds.get_test()
    pred = np.load('models/tbinet/tpreds_ossz_01.npz')['arr_0']
    pred_in = pred[:y_test.shape[0]]
    print('AVG-auc:', avg_auc(y_test[:,:10], pred_in[:,:10]))
    print('AVG-PR-AUC:', avg_psauc(y_test[:,:10], pred_in[:,:10]))
    score = score_fn(pred)
    print(np.sum(y_test))
    # plt.hist(score[:y_test.shape[0], 0])
    # plt.figure()
    # plt.hist(score[y_test.shape[0]:, 0])
    # plt.show()
    is_in = np.zeros((score.shape[0], 1))
    is_in[:y_test.shape[0]] = 1
    # plt.hist(is_in)
    # plt.show()
    print('AUC(in vs shuffle):', avg_auc(is_in, score))
    print('AUPR(in vs shuffle):', avg_psauc(is_in, score))


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
