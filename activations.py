#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class ActivationFunctions:
    def __init__(self):
        self.activation_functions = {
            "sigmoid": (Sigmoid.sigmoid, Sigmoid.sigmoid_prime),
            "softmax": (Softmax.softmax, Softmax.softmax_prime),
            "tanh": (Tanh.tanh, Tanh.tanh_prime),
            "relu": (Relu.relu, Relu.relu_prime),
            "leaky_relu": (LeakyRelu.leaky_relu, LeakyRelu.leaky_relu_prime),
            "elu": (Elu.elu, Elu.elu_prime),
            "gelu": (Gelu.gelu, Gelu.gelu_prime),
        }


class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(z):
        return Sigmoid.sigmoid(z) * (1 - Sigmoid.sigmoid(z))


class Softmax:
    def __init__(self):
        pass

    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z))

    def softmax_prime(z):
        res = []
        for i in range(z.shape[1]):
            c = z[:, i : i + 1]
            j = Softmax.softmax(c) * (np.eye(c.shape[0]) - Softmax.softmax(c).T)
            res.append(j)
        return np.array(res)


class Tanh:
    def __init__(self):
        pass

    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def tanh_prime(z):
        return 1 - np.square(Tanh.tanh(z))


class Relu:
    def __init__(self):
        pass

    def relu(z):
        return np.maximum(0, z)

    def relu_prime(z):
        return np.where(z > 0, 1, 0)


class LeakyRelu:
    def __init__(self):
        pass

    def leaky_relu(z, alpha=0.01):
        return np.maximum(alpha * z, z)

    def leaky_relu_prime(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)


class Elu:
    def __init__(self):
        pass

    def elu(z, alpha=0.01):
        return np.where(z >= 0, z, alpha * (np.exp(z) - 1))

    def elu_prime(z, alpha=0.01):
        return np.where(z > 0, 1, Elu.elu(z, alpha) + alpha)


class Gelu:
    def __init__(self):
        pass

    def gelu(z):
        return 0.5 * z * (1 + Tanh.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * (z**3))))

    def gelu_prime(z):
        return z
