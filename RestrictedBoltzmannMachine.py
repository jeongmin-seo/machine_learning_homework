#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007


   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


"""
import sys
import numpy
from sklearn.model_selection import train_test_split


numpy.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3,\
        W=None, hbias=None, vbias=None, numpy_rng=None, n_output=4, U=None, obias=None, output=None):
        
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer
        self.n_output = n_output    # num of units in output layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if U is None:
            a = 1. / n_hidden
            initial_U = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_hidden, n_output)))

            U = initial_U

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0

        if obias is None:
            obias = numpy.zeros(n_output)   # initialize o bias 0

        self.numpy_rng = numpy_rng
        self.input = input
        self.output = output
        self.W = W  # visual to hidden weight
        self.U = U  # hidden to output weight
        self.hbias = hbias  # hidden bias
        self.vbias = vbias  # visual bias
        self.obias = obias  # output bias

        # self.params = [self.W, self.hbias, self.vbias]

    def contrastive_divergence(self, lr=0.1, k=1, momentum=0.7):

        for i, dat, cls in enumerate(self.input, self.output):
        
            ''' CD-k '''
            ph_prop, ph_sample = self.sample_h_given_x_y(dat, cls)

            chain_start = ph_sample

            for step in xrange(k):
                if step == 0:
                    nv_prop, nv_samples,\
                    nh_prop, nh_samples,\
                    no_prop, no_samples = self.gibbs_hvh(chain_start)
                else:
                    nv_prop, nv_samples,\
                    nh_prop, nh_samples, \
                    no_prop, no_samples = self.gibbs_hvh(nh_samples)

            self.W += lr * ((numpy.dot(dat/self.input.var(axis=0)).T, ph_prop) -
                            numpy.dot(nv_samples / self.input.var(axis=0)).T, nh_prop)  # 모멘트 텀 반영안됨
            self.U += lr * (numpy.dot(ph_prop.T, cls) - numpy.dot(nh_prop, no_samples))
            self.vbias += lr * ((dat - nv_samples)/ self.input.var(axis=0))
            self.hbias += lr * (ph_prop - nh_prop)
            self.obias += lr * numpy.mean(cls - no_samples)


    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_x_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_x_y(v1_sample)
        y1_mean, y1_sample = self.sample_y_given_h(h0_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample,
                y1_mean, y1_sample]
    

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy


    # 직접코딩
    def sample_h_given_x_y(self, x0_sample, y0_sample):
        h1_prop = sigmoid(numpy.dot(x0_sample/self.input.var(axis=0), self.W)+
                          numpy.dot(y0_sample, self.U)+self.hbias)
        h1_sample = self.numpy_rng.binomial(size=h1_prop.shape,n=1,p=h1_prop)

        return [h1_prop, h1_sample]

    def sample_x_given_h(self, h0_sample):
        v1_prop = numpy.dot(h0_sample, self.W.T) + self.vbias
        v1_sample = self.numpy_rng.normal(v1_prop, self.input.std(axis=0), size=v1_prop.shape)

        return [v1_prop, v1_sample]

    def sample_y_given_h(self, h0_sample):
        temp = numpy.dot(h0_sample, self.U) + self.obias
        mean = []
        for tmp in temp:
            mean.append(tmp/temp.sum())

        y1_prop = numpy.asarray(mean)
        y1_sample = self.numpy_rng.multinomial(y1_prop, size=y1_prop.shape, p=y1_prop)

        return [y1_prop, y1_sample]


def data_load(_file_name):

    file_path = "D:\\workspace\\github\\machine_learning_homework\\" + _file_name + ".txt"
    f = open(file_path)
    data = []
    label = []
    if _file_name == 'pima':
        for line in f.readlines():
            split_line = line.split[',']

            label.append(int(split_line[-1]))

            for dat in split_line[0:-1]:
                data.append(float(dat))

    elif _file_name == 'new_tyroid':
        for line in f.readlines():
            split_line = line.split[',']

            label.append(int(split_line[0]))

            for dat in split_line[1:]:
                data.append(float(dat))

    data = numpy.asarray(data)
    label = numpy.asarray(label)

    return data, label

def one_hot_encoding(_label):
    n_class = len(set(_label))
    one_hot_label = []
    for lb in _label:
        tmp = [0] * n_class
        tmp[lb-1] = 1
        one_hot_label.append(tmp)

    one_hot_label = numpy.asarray(one_hot_label)

    return one_hot_label


def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):

    file_name = 'pima'

    data, label = data_load(file_name)
    one_hot_label = one_hot_encoding(label)



    if file_name == 'pima':
        input_node = 8
        hidden_node = 5

    elif file_name == 'new_tyroid':
        input_node = 5
        hidden_node = 2

    rng = numpy.random.RandomState(72170233)

    # construct RBM
    rbm = RBM(input=data, n_visible=input_node, n_hidden=hidden_node, numpy_rng=rng)

    # train
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost





if __name__ == "__main__":
    test_rbm()
