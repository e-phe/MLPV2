#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class ActivationFunctions:
    def __init__(self):
        self.activation_functions = {
            "sigmoid": Sigmoid,
            "softmax": Softmax,
            "tanh": Tanh,
            "relu": Relu,
            "leaky_relu": LeakyRelu,
            "elu": Elu,
            "gelu": Gelu,
        }

    @staticmethod
    def last_layer(activation, zs, loss, delta_activation):
        if isinstance(activation, Softmax):
            delta = delta_activation @ loss.T.reshape(
                zs[-1].shape[1], zs[-1].shape[0], 1
            )
            return delta[:, :, 0].T
        else:
            return loss * delta_activation


class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class Softmax:
    def __init__(self):
        pass

    @staticmethod
    def activation(z):
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def prime(z):
        res = []
        for i in range(z.shape[1]):
            c = z[:, i : i + 1]
            j = Softmax.activation(c) * (np.eye(c.shape[0]) - Softmax.activation(c).T)
            res.append(j)
        return np.array(res)


class Tanh:
    def __init__(self):
        pass

    @staticmethod
    def activation(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def prime(z):
        return 1 - np.square(Tanh.activation(z))


class Relu:
    def __init__(self):
        pass

    @staticmethod
    def activation(z):
        return np.maximum(0, z)

    @staticmethod
    def prime(z):
        return np.where(z > 0, 1, 0)


class LeakyRelu:
    def __init__(self):
        pass

    @staticmethod
    def activation(z, alpha=0.01):
        return np.maximum(alpha * z, z)

    @staticmethod
    def prime(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)


class Elu:
    def __init__(self):
        pass

    @staticmethod
    def activation(z, alpha=0.01):
        return np.where(z >= 0, z, alpha * (np.exp(z) - 1))

    @staticmethod
    def prime(z, alpha=0.01):
        return np.where(z > 0, 1, Elu.activation(z, alpha) + alpha)


class Gelu:
    def __init__(self):
        pass

    @staticmethod
    def activation(z):
        return (
            0.5
            * z
            * (1 + Tanh.activation(np.sqrt(2 / np.pi) * (z + 0.044715 * (z**3))))
        )

    @staticmethod
    def prime(z):
        return z
