#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import UsefulFunction
from utils import Range
from network import Network
import argparse
import numpy as np


def parser():
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "-dataset_train",
        "-train",
        help="path of the training dataset",
    )
    parser.add_argument(
        "-dataset_evaluation",
        "-eval",
        help="path of the evaluation dataset",
    )
    parser.add_argument(
        "-models",
        "-m",
        help="load models from file",
    )
    parser.add_argument(
        "-layers",
        "-l",
        nargs="+",
        type=int,
        help="number of layers on each layer",
        default=[784, 256, 128, 10],
    )
    parser.add_argument(
        "-alpha",
        "-a",
        type=float,
        help="learning rate",
        default=1.0,
    )
    parser.add_argument(
        "-epochs",
        "-e",
        type=int,
        help="number of iterations",
        default=10,
    )
    parser.add_argument(
        "-mini_batch_size",
        "-mb",
        type=int,
        help="size of the mini batches",
        default=32,
    )
    parser.add_argument(
        "-xavier",
        "-x",
        action="store_true",
        help="use Xavier weight initialization",
        default=False,
    )
    parser.add_argument(
        "-loss_functions",
        "-lf",
        choices=["binary_cross_entropy, categorical_cross_entropy"],
        help="choose loss function",
        default="categorical_cross_entropy",
    )
    parser.add_argument(
        "-activation_functions",
        "-af",
        choices=["sigmoid", "softmax", "tanh", "relu", "leaky_relu", "elu", "gelu"],
        nargs="+",
        help="choose activation functions",
        default=None,
    )
    parser.add_argument(
        "-batch_normalization",
        "-bn",
        choices=["none", "before", "after"],
        nargs="+",
        help="apply batch normalization before or after the activation function",
        default=None,
    )
    parser.add_argument(
        "-dropout",
        "-d",
        choices=[Range(0, 1)],
        type=float,
        nargs="+",
        help="choose keep rate, probability of dropout",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    kwargs = parser()
    ufunc = UsefulFunction()

    if kwargs.dataset_train:
        try:
            dataset_train = np.loadtxt(
                kwargs.dataset_train,
                delimiter=",",
                skiprows=1,
            )
        except:
            exit(f"FileNotFoundError: Can't find {kwargs.dataset_train}")
        x = [ufunc.normalization(np.reshape(x, (-1, 1))) for x in dataset_train[:, :-1]]
        y = [ufunc.vectorized_result(y, 10) for y in dataset_train[:, [-1]]]
        x_train, x_validation, y_train, y_validation = ufunc.data_spliter(x, y, 0.7)
    else:
        x_train, x_validation, y_train, y_validation = [], [], [], []

    param = {
        "layers": kwargs.layers,
        "alpha": kwargs.alpha,
        "epochs": kwargs.epochs,
        "mini_batch_size": kwargs.mini_batch_size,
        "xavier": kwargs.xavier,
        "loss_functions": kwargs.loss_functions,
        "activation_functions": kwargs.activation_functions,
        "batch_normalization": kwargs.batch_normalization,
        "dropout": kwargs.dropout,
    }

    net = Network(
        list(zip(x_train, y_train)),
        list(zip(x_validation, y_validation)),
        **{k: v for k, v in param.items() if v is not None},
    )

    if kwargs.models:
        try:
            models = np.load(kwargs.models, allow_pickle=True)
            biases = models["biases"]
            weights = models["weights"]
        except:
            exit(f"FileNotFoundError: Can't find {kwargs.models}")
    else:
        if kwargs.dataset_train:
            (biases, weights) = net.gradient_descent()
            np.savez("models", biases=biases, weights=weights)

    if kwargs.dataset_evaluation:
        try:
            dataset_eval = np.loadtxt(
                kwargs.dataset_evaluation,
                delimiter=",",
                skiprows=1,
            )
            x_eval = [ufunc.normalization(np.reshape(x, (-1, 1))) for x in dataset_eval]
            net.compute_answer(x_eval, biases, weights)
        except:
            exit(f"FileNotFoundError: Can't find {kwargs.dataset_evaluation}")
