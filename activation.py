#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class ActivationFunction:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def leaky_relu(self, z, alpha=0.01):
        return np.maximum(alpha * z, z)

    def elu(self, z, alpha=0.01):
        return np.where(z >= 0, z, alpha * (np.exp(z) - 1))

    def gelu(self, z):
        return 0.5 * z * (1 + self.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * (z**3))))


class ActivationFunctionPrime:
    def __init__(self):
        self.af = ActivationFunction()

    def sigmoid_prime(self, z):
        return self.af.sigmoid(z) * (1 - self.af.sigmoid(z))

    def softmax_prime(self, z):
        res = []
        for i in range(z.shape[1]):
            c = z[:, i : i + 1]
            j = self.af.softmax(c) * (np.eye(c.shape[0]) - self.af.softmax(c).T)
            res.append(j)
        return np.array(res)

    def tanh_prime(self, z):
        return 1 - np.square(self.af.tanh(z))

    def relu_prime(self, z):
        return np.where(z > 0, 1, 0)

    def leaky_relu_prime(self, z, alpha=0.01):
        return np.where(z > 0, 1, alpha)

    def elu_prime(self, z, alpha=0.01):
        return np.where(z > 0, 1, self.af.elu(z, alpha) + alpha)

    def gelu_prime(self, z):
        return z
