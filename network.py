#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from activations import ActivationFunctions
from batchnorm import Batchnorm
from loss import LossFunctions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Network:
    def __init__(
        self,
        training_data,
        validation_data,
        layers,
        alpha,
        epochs,
        mini_batch_size,
        xavier,
        loss_functions,
        activation_functions=None,
        batch_normalization=None,
        dropout=None,
    ):
        if (
            isinstance(layers, list)
            and len(layers)
            and alpha
            and isinstance(alpha, float)
            and epochs
            and isinstance(epochs, int)
            and mini_batch_size
            and isinstance(mini_batch_size, int)
            and loss_functions
            and isinstance(loss_functions, str)
            and isinstance(xavier, bool)
        ):
            self.training_data = training_data
            self.validation_data = validation_data
            self.layers = layers
            self.num_layers = len(layers)
            self.alpha = alpha
            self.epochs = epochs
            self.mini_batch_size = mini_batch_size
        else:
            exit("Error: Bad initialization value")

        if xavier:
            self.biases = [np.random.randn(y, 1) for y in layers[1:]]
            self.weights = [
                np.random.randn(y, x) * (1 / np.sqrt(x))
                for x, y in zip(layers[:-1], layers[1:])
            ]
        else:
            self.biases = [np.random.randn(y, 1) for y in layers[1:]]
            self.weights = [
                np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])
            ]
        if activation_functions == None:
            activation_functions = ["sigmoid"] * self.num_layers
            activation_functions[-1] = "softmax"
        if (
            isinstance(activation_functions, list)
            and len(activation_functions) == self.num_layers
        ):
            self.activation_functions = list(
                map(
                    ActivationFunctions().activation_functions.get,
                    activation_functions,
                )
            )
        else:
            exit("Error: Bad activation fonction value")

        if batch_normalization == None:
            self.batch_normalization = [None] * self.num_layers
        elif (
            isinstance(batch_normalization, list)
            and len(batch_normalization) == self.num_layers
        ):
            self.batch_normalization = batch_normalization
        else:
            exit("Error: Bad batch normalization value")
        self.batchnorms = [Batchnorm(i) for i in layers[1:]]

        self.loss_functions = LossFunctions().loss_functions.get(loss_functions)

        if dropout == None:
            self.keep_rate = [1.0] * self.num_layers
        elif isinstance(dropout, list) and len(dropout) == self.num_layers - 1:
            self.keep_rate = dropout
        else:
            exit("Error: Bad dropout value")

    def gradient_descent(self):
        training = np.zeros(self.epochs * 2).reshape(self.epochs, 2)
        validation = np.zeros(self.epochs * 2).reshape(self.epochs, 2)
        learning = self.alpha / self.mini_batch_size
        best = [np.iinfo(np.int32).max, 0]
        biases = []
        weights = []

        for i in range(self.epochs):
            mini_batches = self.create_mini_batch()

            for mini_batch in mini_batches:
                x, y = self.split_xy(mini_batch)

                zs, activations, dropout = self.feed_forward(x)
                nabla_b, nabla_w = self.back_propagation(x, y, zs, activations, dropout)
                nabla_b = [np.mean(nb, axis=1).reshape(-1, 1) for nb in nabla_b]

                self.biases -= learning * np.array(nabla_b)
                self.weights -= learning * np.array(nabla_w)

            training[i] = self.evaluate(self.training_data)
            validation[i] = self.evaluate(self.validation_data)
            if best[0] > validation[i, 0] and best[1] < validation[i, 1]:
                best = validation[i]
                biases = self.biases
                weights = self.weights
            print(
                "epoch {}/{} - loss: {} - validation_loss: {}\naccuracy: {} - validation_accuracy: {}".format(
                    i,
                    self.epochs,
                    training[i, 0],
                    validation[i, 0],
                    training[i, 1],
                    validation[i, 1],
                )
            )
        self.display_graph(training, validation)
        if not len(biases) or not len(weights):
            return (self.biases, self.weights)
        return (biases, weights)

    def create_mini_batch(self):
        n = len(self.training_data)
        np.random.shuffle(self.training_data)
        return [
            self.training_data[k : k + self.mini_batch_size]
            for k in range(0, n, self.mini_batch_size)
        ]

    def split_xy(self, mini_batch):
        x, y = zip(*mini_batch)
        return (
            np.asarray(x).reshape(-1, len(x[0])).T,
            np.asarray(y).reshape(-1, len(y[0])).T,
        )

    def feed_forward(self, x):
        activation = x
        activations = [x]
        zs = []
        dropout = []
        for i in range(self.num_layers - 1):
            z = self.weights[i] @ activation + self.biases[i]
            if self.batch_normalization[i] == "before":
                z = self.batchnorms[i].batch_norm_forward(z)
            zs.append(z)
            activation = self.activation_functions[i].activation(z)
            if self.batch_normalization[i] == "after":
                activation = self.batchnorms[i].batch_norm_forward(activation)
            dropout.append(
                np.random.rand(activation.shape[0], activation.shape[1])
                < self.keep_rate[i]
            )
            activation = activation * dropout[i] / self.keep_rate[i]
            activations.append(activation)
        return (zs, activations, dropout)

    def back_propagation(self, x, y, zs, activations, dropout):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        loss = self.loss_functions.prime(y, activations[-1])
        delta_activation = self.activation_functions[-1].prime(zs[-1])
        delta = ActivationFunctions.last_layer(
            self.activation_functions[-1](), zs, loss, delta_activation
        )

        for l in range(2, self.num_layers):
            z = zs[-l]
            if self.batch_normalization[-l] == "after":
                z = self.batchnorms[-l].batch_norm_backward(z, x, self.alpha)
            delta_activation = self.activation_functions[-l].prime(z)
            if self.batch_normalization[-l] == "before":
                delta_activation = self.batchnorms[-l].batch_norm_backward(
                    delta_activation, x, self.alpha
                )
            delta_activation = delta_activation * dropout[-l] / self.keep_rate[-l]
            delta = (self.weights[-l + 1].T @ delta) * delta_activation

            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l - 1].T
        return (nabla_b, nabla_w)

    def evaluate(self, data):
        x, y = self.split_xy(data)
        y = y.T
        y_hat = self.predict(x, self.biases, self.weights).T
        loss = self.loss_functions.loss(y.reshape(-1, 1), y_hat.reshape(-1, 1))
        accuracy = self.accuracy(y, y_hat)
        return [loss, accuracy]

    def predict(self, activation, biases, weights):
        for i in range(self.num_layers - 1):
            z = weights[i] @ activation + biases[i]
            if self.batch_normalization[i] == "before":
                z = self.batchnorms[i].batch_norm(z)
            activation = self.activation_functions[i].activation(z)
            if self.batch_normalization[i] == "after":
                activation = self.batchnorms[i].batch_norm(activation)
        return activation

    def accuracy(self, y, y_hat):
        results = [(np.argmax(Y), np.argmax(Y_hat)) for (Y, Y_hat) in zip(y, y_hat)]
        return sum(int(x == y) for (x, y) in results) / len(y)

    def display_graph(self, training, validation):
        x = range(self.epochs)
        y_label = ["loss", "accuracy"]
        figure, axis = plt.subplots(1, 2)
        for i in range(2):
            axis[i].set_xlabel("epochs")
            axis[i].set_ylabel(y_label[i])
            axis[i].scatter(x, training[:, i], label="training")
            axis[i].scatter(x, validation[:, i], label="validation")
            axis[i].plot(x, training[:, i], label="training")
            axis[i].plot(x, validation[:, i], label="validation")
            axis[i].legend()
        plt.show()

    def compute_answer(self, a, biases, weights):
        a = self.predict(a, biases, weights)
        results = np.array([np.argmax(y_hat) for y_hat in a]).reshape(-1, 1)
        pd.DataFrame(results).rename(columns={0: "label"}).to_csv(
            "my_answer.csv", index_label="rowid"
        )
