#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class LossFunctions:
    def __init__(self):
        self.loss_functions = {
            "binary_cross_entropy": BinaryCrossEntropy,
            "categorical_cross_entropy": CategoricalCrossEntropy,
        }


class BinaryCrossEntropy:
    def __init__(self):
        pass

    @staticmethod
    def loss(y, y_hat, eps=1e-15):
        return (
            -np.sum(y.T @ np.log(y_hat + eps) + (1 - y).T @ np.log(1 - y_hat + eps))
            / y.shape[0]
        )

    @staticmethod
    def prime(y, y_hat):
        return y_hat - y


class CategoricalCrossEntropy:
    def __init__(self):
        pass

    @staticmethod
    def loss(y, y_hat, eps=1e-15):
        return -np.sum(y.T @ np.log(y_hat + eps)) / y.shape[0]

    @staticmethod
    def prime(y, y_hat, eps=1e-15):
        return -y / (y_hat + eps)
