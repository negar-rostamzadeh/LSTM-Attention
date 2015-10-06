import theano.tensor as T

from blocks.bricks.base import application

import bricks
import initialization

import util
import masonry

class SingleSoftmax(bricks.Initializable):
    def __init__(self, input_dim, n_classes, batch_normalize, **kwargs):
        super(SingleSoftmax, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.n_classes = n_classes

        self.mlp = masonry.construct_mlp(
            name="mlp",
            activations=[None, bricks.Identity()],
            input_dim=input_dim,
            hidden_dims=[input_dim/2, self.n_classes],
            batch_normalize=batch_normalize,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0))
        self.softmax = bricks.Softmax()

        self.children = [self.mlp, self.softmax]

    @application(inputs=['x', 'y'], outputs=['cost'])
    def cost(self, x, y, n_patches):
        energy = self.mlp.apply(x)
        cross_entropy = self.softmax.categorical_cross_entropy(
            y.flatten(), energy).mean(axis=0)
        error_rate = T.neq(y, energy.argmax(axis=1)).mean(axis=0)
        cost = util.named(cross_entropy, "cost")
        self.add_auxiliary_variable(cross_entropy, name="cross_entropy")
        self.add_auxiliary_variable(error_rate, name="error_rate")
        return cost
