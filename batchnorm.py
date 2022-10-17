#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class Batchnorm:
    def __init__(self, size):
        self.eps = 1e-4
        self.gamma = np.ones((size, 1))
        self.beta = np.zeros((size, 1))
        self.moving_mean = np.zeros((size, 1))
        self.moving_var = np.zeros((size, 1))

        self.batch_mean = []
        self.batch_var = []
        self.std = []
        self.x_centered = []
        self.x_norm = []
        self.xhat = []

    def batch_norm_forward(self, x, momentum=0.9):
        self.batch_mean = np.mean(x, axis=1).reshape(-1, 1)
        self.batch_var = np.var(x, axis=1).reshape(-1, 1)
        self.std = np.sqrt(self.batch_var + self.eps)

        self.moving_mean = (
            momentum * self.moving_mean + (1 - momentum) * self.batch_mean
        )
        self.moving_var = momentum * self.moving_var + (1 - momentum) * self.batch_var

        self.x_centered = x - self.batch_mean
        self.x_norm = self.x_centered / self.std
        self.xhat = self.gamma * self.x_norm + self.beta
        return self.xhat

    def batch_norm_backward(self, dout, x, alpha):
        dgamma = np.sum(dout * self.x_norm, axis=1).reshape(-1, 1)
        dbeta = np.sum(dout, axis=1).reshape(-1, 1)

        dx_norm = dout * self.gamma
        dx_centered = dx_norm / self.std
        dmean = -(
            np.sum(dx_centered, axis=0)
            + 2 / x.shape[0] * np.sum(self.x_centered, axis=0)
        )
        dstd = np.sum(dx_norm * self.x_centered * -(1 / self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dx = dx_centered + (dmean + dvar * 2 * self.x_centered) / x.shape[0]

        self.gamma = self.gamma - alpha * dgamma
        self.beta = self.beta - alpha * dbeta
        return dx

    def batch_norm(self, x):
        self.x_norm = (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
        return self.gamma * self.x_norm + self.beta
