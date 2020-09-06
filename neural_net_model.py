import pickle

import numpy as np
import copy


# ================================================
import data_loader as dl


class CrossEntropyLoss(object):

    @staticmethod
    def func(a, y):
        # a = softmax(a)
        return [np.sum(np.nan_to_num(-y_*np.log(a_)-(1-y_)*np.log(1-a_))) for (a_, y_) in zip(a.T, y.T)]

    @staticmethod
    def derivative(a, y):
        return a - y


class Sigmoid(object):

    @staticmethod
    def func(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.func(z) * (1 - Sigmoid.func(z))


class ReLU(object):

    @staticmethod
    def func(z):
        t = copy.deepcopy(z)
        t[t < 0] = 0
        return t

    @staticmethod
    def prime(z):
        t = copy.deepcopy(z)
        t[t <= 0] = 0
        t[t > 0] = 1
        return t


def softmax(a):
    exp = np.exp(a - np.max(a))
    return exp / exp.sum(0)


# =====================================================


class NN:
    """
    Generic fully connected neural network
    """

    def __init__(self, net_shape, loss, activation_f, last_activation_f):
        """
        Constructor
        :param net_shape: array with num of neurons in each layer
        :param loss: loss function
        :param activation_f: activation function in all but the last layer
        :param last_activation_f: activation function in the last layer
        """
        self.num_of_layers = len(net_shape)
        self.weights = self.init_weights(net_shape)  # each element is a matrix of weights between layers
        self.biases = self.init_biases(net_shape)  # each element is a vector of biases of a layer
        self.loss = loss
        self.activation_f = activation_f
        self.last_activation_f = last_activation_f

    def init_weights(self, net_shape):
        """
        initialize weights matrices
        :param net_shape: shape of the net
        :return: weights initialized randomly
        """
        return [np.random.randn(n_in, n_out) * np.sqrt(1.0/n_out)
                for (n_out, n_in) in zip(net_shape[:-1], net_shape[1:])]

    def init_biases(self, net_shape):
        """
        initialize weights vectors
        :param net_shape: shape of the net
        :return: biases initialized with zeroes
        """
        return [np.zeros((n_in, 1)) for n_in in net_shape[1:]]

    def forward_pass(self, examples):
        """
        run given examples through the net and store each layers' input and output val
        :param examples: given examples
        :return: z_mat: values inserted to a layer (z=Wx+b), activations_mat: values leaving the layer (f(Wx+b))
        """

        curr_activation = examples
        z_mat, activations_mat = [], [examples]
        i = 0
        for (w, b) in zip(self.weights, self.biases):
            z = np.dot(w, curr_activation)+b
            z_mat.append(z)
            if i == self.num_of_layers - 2:
                curr_activation = self.last_activation_f.func(z)
            else:
                curr_activation = self.activation_f.func(z)
            activations_mat.append(curr_activation)
            i += 1
        return z_mat, activations_mat

    def back_propagation(self, labels, z_mat, activations_mat):
        """
        backprop algorithm to propagate the error backwards in order to compute the gradient
        of the loss with respect to w and b
        :param labels: true labels of the given examples
        :param z_mat: array with values inserted to a layer (z=Wx+b) for every layer
        :param activations_mat: result (activation) of each layer
        :return: gradient of the loss with respect to w and b, the loss and accuracy of the run on the batch
        """
        partial_by_w, partial_by_b = [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
        loss = np.mean(self.loss.func(activations_mat[-1], labels))
        results = [(np.argmax(a), np.argmax(label)) for (a, label) in zip(activations_mat[-1].T, labels.T)]
        accuracy = np.count_nonzero([a == b for (a, b) in results]) / len(results)

        delta = self.loss.derivative(activations_mat[-1], labels) * self.last_activation_f.prime(z_mat[-1])
        partial_by_w[-1] = np.dot(delta, activations_mat[-2].T)
        partial_by_b[-1] = delta.sum(1).reshape([len(delta), 1])
        for lay_num in range(2, self.num_of_layers):
            delta = np.dot(self.weights[-lay_num+1].T, delta) * self.activation_f.prime(z_mat[-lay_num])
            partial_by_w[-lay_num] = np.dot(delta, activations_mat[-lay_num - 1].T)
            partial_by_b[-lay_num] = delta.sum(1).reshape([len(delta), 1])
        return partial_by_w, partial_by_b, loss, accuracy

    def back_prop_batch(self, mini_batch):
        """
        apply the back propagation algorithm on the given mini batch
        :param mini_batch: given mini batch
        :return: gradient of the loss with respect to w and b, the loss and accuracy of the run on the batch
        """
        examples, labels = mini_batch[0][0], mini_batch[0][1]
        for example, label in mini_batch[1:]:
            examples, labels = np.concatenate((examples, example), axis=1), np.concatenate((labels, label), axis=1)
        # forward_pass the batch
        z_mat, activations_mat = self.forward_pass(examples)
        # back prop the batch
        partial_by_w, partial_by_b, loss, accuracy = self.back_propagation(labels, z_mat, activations_mat)
        return partial_by_w, partial_by_b, loss, accuracy

    def SGD(self, training_data, num_of_epochs, mini_batch_size, reg, eta):
        """
        stochastic gradient descent algorithm to update the weights and biases of the network by moving
        in the direction that minimizes the loss function
        :param training_data: given training data (examples and labels)
        :param num_of_epochs: amount of epochs to iterate
        :param mini_batch_size: size of the mini batch
        :param reg: regularization hyper-parameter
        :param eta: learning rate hyper-parameter
        :return:
        """
        len_all_training = len(training_data)
        for epoch in range(num_of_epochs):
            if epoch > 0 and (epoch+1) % 5 == 0:
                eta = eta * 0.5
            np.random.shuffle(training_data)
            mini_batches = [training_data[j: j+mini_batch_size] for j in range(0, len_all_training, mini_batch_size)]
            batches_loss, batches_accuracies = [], []
            for mini_batch in mini_batches:
                partial_by_w, partial_by_b, mini_batch_loss, mini_batch_accuracy = self.back_prop_batch(mini_batch)
                batches_loss.append(mini_batch_loss)
                batches_accuracies.append(mini_batch_accuracy)
                self.weights = [(1-eta*(reg/len_all_training)) * w - (eta/len(mini_batch)) * par_w
                                for (w, par_w) in zip(self.weights, partial_by_w)]
                self.biases = [b - (eta/len(mini_batch))*par_b for (b, par_b) in zip(self.biases, partial_by_b)]
            loss = np.mean(batches_loss)
            accuracy = np.mean(batches_accuracies)
            print("epoch {} complete. train loss: {}, train accuracy: {}".format(epoch, loss, accuracy))
        return

    def train(self, training_data, num_of_epochs, mini_batch_size, reg, eta):
        """
        trianing the neural net using SGD
        :param training_data: given training data (examples and labels)
        :param num_of_epochs: amount of epochs to iterate
        :param mini_batch_size: size of the mini batch
        :param reg: regularization hyper-parameter
        :param eta: learning rate hyper-parameter
        :return:
        """
        self.SGD(training_data, num_of_epochs, mini_batch_size, reg, eta)

    def feed_forward(self, a):
        """
        evaluate the network response for a given input
        :param a: given input to feed the nn
        :return: the response of the nn (values in the last layer)
        """
        for (w, b) in zip(self.weights, self.biases):
            a = self.activation_f.func(np.dot(w, a) + b)
        return a

    def predict(self, samples):
        """
        class probabilities predictions of the net on given array of samples
        :param samples: array of examples to feed the net
        :return: class probabilities predictions
        """
        return [softmax(self.feed_forward(x)) if not np.all(x==1) else None for x in samples ]

    def score(self, data):
        """
        score of the model on given data
        :param data: data to test the model on (samples and labels)
        :return: accuracy of the predictions of the net on the given data
        """
        results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in data]
        return np.count_nonzero([a == b for (a, b) in results]) / len(results)


def save_to_pkl(model, name):
    file = open(name, 'wb')
    pickle.dump(model, file)


def vote_predict(model,images):
    """

    :type model: NM
    """
    p = np.array(model.predict(images))
    p_sum = np.sum(p, axis=0)
    return np.argmax(p_sum)
