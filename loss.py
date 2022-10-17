#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class LossFunctions:
    def __init__(self, eps=1e-15):
        self.eps = eps
        self.loss_functions = {
            "binary_cross_entropy": (
                BinaryCrossEntropy.binary_cross_entropy,
                BinaryCrossEntropy.binary_cross_entropy_prime,
            ),
            "categorical_cross_entropy": (
                CategoricalCrossEntropy.categorical_cross_entropy,
                CategoricalCrossEntropy.categorical_cross_entropy_prime,
            ),
        }


class BinaryCrossEntropy:
    def __init__(self):
        pass

    def binary_cross_entropy(y, y_hat):
        return (
            -np.sum(
                y.T @ np.log(y_hat + LossFunctions().eps)
                + (1 - y).T @ np.log(1 - y_hat + LossFunctions().eps)
            )
            / y.shape[0]
        )

    def binary_cross_entropy_prime(y, y_hat):
        return y_hat - y


class CategoricalCrossEntropy:
    def __init__(self):
        pass

    def categorical_cross_entropy(y, y_hat):
        return -np.sum(y.T @ np.log(y_hat + LossFunctions().eps)) / y.shape[0]

    def categorical_cross_entropy_prime(y, y_hat):
        return -y / (y_hat + LossFunctions().eps)
