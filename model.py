import logging
import theano
import theano.tensor as T
from blocks.bricks.cost import CategoricalCrossEntropy, SquaredError
from blocks.bricks import MLP, Linear, Tanh, Softmax
from blocks.bricks.recurrent import LSTM
from blocks.initialization import IsotropicGaussian, Constant
logger = logging.getLogger('main.model')
floatX = theano.config.floatX


class MLPModel():
    def __init__(self, name='MLP'):
        self.non_lins = [Tanh(), Softmax()]
        self.dims = [784, 100, 10]
        self.default_lr = 0.1
        self.name = name

    def apply(self, input_, target):
        mlp = MLP(self.non_lins, self.dims,
                  weights_init=IsotropicGaussian(0.01),
                  biases_init=Constant(0),
                  name=self.name)
        mlp.initialize()
        probs = mlp.apply(T.flatten(input_, outdim=2))
        probs.name = 'probs'
        cost = CategoricalCrossEntropy().apply(target.flatten(), probs)
        cost.name = "CE"
        self.outputs = {}
        self.outputs['probs'] = probs
        self.outputs['cost'] = cost


class LSTMModel():
    def __init__(self, name='LSTM'):
        self.dims = [2, 7, 2]
        self.default_lr = 0.01
        self.name = name

    def apply(self, input_, target):
        x_to_h = Linear(name='x_to_h',
                        input_dim=self.dims[0],
                        output_dim=self.dims[1] * 4)
        pre_rnn = x_to_h.apply(input_)
        pre_rnn.name = 'pre_rnn'
        rnn = LSTM(activation=Tanh(),
                   dim=self.dims[1], name=self.name)
        h, _ = rnn.apply(pre_rnn)
        h.name = 'h'
        h_to_y = Linear(name='h_to_y',
                        input_dim=self.dims[1],
                        output_dim=self.dims[2])
        y_hat = h_to_y.apply(h)
        y_hat.name = 'y_hat'

        cost = SquaredError().apply(target, y_hat)
        cost.name = 'MSE'

        self.outputs = {}
        self.outputs['y_hat'] = y_hat
        self.outputs['cost'] = cost
        self.outputs['pre_rnn'] = pre_rnn
        self.outputs['h'] = h

        # Initialization
        for brick in (rnn, x_to_h, h_to_y):
            brick.weights_init = IsotropicGaussian(0.01)
            brick.biases_init = Constant(0)
            brick.initialize()
