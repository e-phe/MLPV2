#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class LossFunction:
    def __init__(self):
        self.eps = 1e-15

    def cross_entropy(self, y, y_hat):
        return -np.sum(y.T @ np.log(y_hat + self.eps)) / y.shape[0]

    def cross_entropy_prime(self, y, y_hat):
        return y_hat - y
