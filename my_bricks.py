import operator

import theano.tensor as T

from blocks.bricks.base import application, lazy
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans

import blocks_bricks as bricks
import initialization

class NormalizedActivation(bricks.Initializable, bricks.Feedforward):
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, activation=None, batch_normalize=False, **kwargs):
        super(NormalizedActivation, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable
        self.activation = activation or bricks.Rectifier()
        self.batch_normalize = batch_normalize

    @property
    def broadcastable(self):
        return self._broadcastable or [False]*len(self.shape)

    @broadcastable.setter
    def broadcastable(self, broadcastable):
        self._broadcastable = broadcastable

    def _allocate(self):
        arghs = dict(shape=self.shape,
                     broadcastable=self.broadcastable)
        sequence = []
        if self.batch_normalize:
            sequence.append(Standardization(**arghs))
            sequence.append(SharedScale(
                weights_init=initialization.Constant(1),
                **arghs))
        sequence.append(SharedShift(
            biases_init=initialization.Constant(0),
            **arghs))
        sequence.append(self.activation)
        self.sequence = bricks.FeedforwardSequence([
            brick.apply for brick in sequence
        ], name="ffs")
        self.children = [self.sequence]

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        return self.sequence.apply(input_)

    def get_dim(self, name):
        try:
            return dict(input_=self.shape,
                        output=self.shape)
        except:
            return super(NormalizedActivation, self).get_dim(name)

class FeedforwardFlattener(bricks.Flattener, bricks.Feedforward):
    def __init__(self, input_shape, **kwargs):
        super(FeedforwardFlattener, self).__init__(**kwargs)
        self.input_shape = input_shape

    @property
    def input_dim(self):
        return reduce(operator.mul, self.input_shape)

    @property
    def output_dim(self):
        return reduce(operator.mul, self.input_shape)

class FeedforwardIdentity(bricks.Feedforward):
    def __init__(self, dim, **kwargs):
        super(FeedforwardIdentity, self).__init__(**kwargs)
        self.dim = dim

    @property
    def input_dim(self):
        return self.dim

    @property
    def output_dim(self):
        return self.dim

    @application(inputs=["x"], outputs=["x"])
    def apply(self, x):
        return x

class SharedScale(bricks.Initializable, bricks.Feedforward):
    """
    Element-wise scaling with optional parameter-sharing across axes.
    """
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, **kwargs):
        super(SharedScale, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable

    def _allocate(self):
        parameter_shape = [1 if broadcast else dim
                           for dim, broadcast in zip(self.shape, self.broadcastable)]
        self.gamma = shared_floatx_nans(parameter_shape, name='gamma')
        add_role(self.gamma, WEIGHT)
        self.parameters.append(self.gamma)
        self.add_auxiliary_variable(self.gamma.norm(2), name='gamma_norm')

    def _initialize(self):
        self.weights_init.initialize(self.gamma, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ * T.patternbroadcast(self.gamma, self.broadcastable)

    def get_dim(self, name):
        if name == 'input_':
            return self.shape
        if name == 'output':
            return self.shape
        return super(SharedScale, self).get_dim(name)

class SharedShift(bricks.Initializable, bricks.Feedforward):
    """
    Element-wise bias with optional parameter-sharing across axes.
    """
    @lazy(allocation="shape broadcastable".split())
    def __init__(self, shape, broadcastable, **kwargs):
        super(SharedShift, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable

    def _allocate(self):
        parameter_shape = [1 if broadcast else dim
                           for dim, broadcast in zip(self.shape, self.broadcastable)]
        self.beta = shared_floatx_nans(parameter_shape, name='beta')
        add_role(self.beta, BIAS)
        self.parameters.append(self.beta)
        self.add_auxiliary_variable(self.beta.norm(2), name='beta_norm')

    def _initialize(self):
        self.biases_init.initialize(self.beta, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_ + T.patternbroadcast(self.beta, self.broadcastable)

    def get_dim(self, name):
        if name == 'input_':
            return self.shape
        if name == 'output':
            return self.shape
        return super(SharedShift, self).get_dim(name)

# TODO: replacement of batch/population statistics by annotations
# TODO: depends on replacements inside scan
class Standardization(bricks.Initializable, bricks.Feedforward):
    stats = "mean var".split()

    def __init__(self, shape, broadcastable, alpha=1e-2, **kwargs):
        super(Standardization, self).__init__(**kwargs)
        self.shape = shape
        self.broadcastable = broadcastable
        self.alpha = alpha

    def _allocate(self):
        parameter_shape = [1 if broadcast else dim
                           for dim, broadcast in zip(self.shape, self.broadcastable)]
        self.population_stats = dict(
            (stat, shared_floatx_nans(parameter_shape,
                                      name="population_%s" % stat))
            for stat in self.stats)

    def _initialize(self):
        for stat, initializer in (("mean", 0), ("var",  1)):
            self.population_stats[stat].get_value().fill(initializer)

    @application(inputs=["input_"], outputs=["output"])
    def apply(self, input_):
        aggregate_axes = [0] + [1 + i for i, b in enumerate(self.broadcastable) if b]
        self.batch_stats = dict(
            (stat, getattr(input_, stat)(axis=aggregate_axes,
                                         keepdims=True)[0])
            for stat in self.stats)

        # NOTE: these are unused for now
        self._updates = [(self.population_stats[stat],
                          (1 - self.alpha)*self.population_stats[stat]
                          + self.alpha*self.batch_stats[stat])
                         for stat in self.stats]
        self._replacements = [(self.batch_stats[stat], self.population_stats[stat])
                              for stat in self.stats]

        return ((input_ - self.batch_stats["mean"])
                / (T.sqrt(self.batch_stats["var"] + 1e-8)))