from blocks.bricks import Initializable, Tanh, Rectifier, MLP
from blocks.bricks.base import application, lazy
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.bricks.recurrent import BaseRecurrent, recurrent
import theano.tensor as tensor
import numpy as np
from crop import LocallySoftRectangularCropper
from crop import Gaussian


class LSTMAttention(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, mlp_hidden_dims, batch_size,
                 image_shape, patch_shape, activation=None, **kwargs):
        super(LSTMAttention, self).__init__(**kwargs)
        self.dim = dim
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        non_lins = [Rectifier()] * (len(mlp_hidden_dims) - 1) + [Tanh()]
        mlp_dims = [np.prod(patch_shape) + dim + 4] + mlp_hidden_dims
        mlp = MLP(non_lins, mlp_dims,
                  weights_init=self.weights_init,
                  biases_init=self.biases_init,
                  name=self.name + '_mlp')
        hyperparameters = {}
        hyperparameters["cutoff"] = 3
        hyperparameters["batched_window"] = True
        cropper = LocallySoftRectangularCropper(
            patch_shape=patch_shape,
            hyperparameters=hyperparameters,
            kernel=Gaussian())
        self.rescaling_factor = patch_shape[0] / image_shape[0]
        self.min_scale = self.rescaling_factor

        if not activation:
            activation = Tanh()
        self.children = [activation, mlp, cropper]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name in ['location', 'scale']:
            return 2
        if name == 'mask':
            return 0
        return super(LSTMAttention, self).get_dim(name)

    def _allocate(self):
        self.W_patch = shared_floatx_nans((np.prod(self.patch_shape) + 4,
                                           4 * self.dim),
                                          name='W_input')
        self.W_state = shared_floatx_nans((self.dim, 4 * self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        self.initial_location = shared_floatx_zeros((2,),
                                                    name="initial_location")
        self.initial_scale = shared_floatx_zeros((2,),
                                                 name="initial_scale")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_patch, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)
        add_role(self.initial_location, INITIAL_STATE)
        add_role(self.initial_scale, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
            self.W_patch, self.W_cell_to_out, self.initial_state_,
            self.initial_cells, self.initial_location, self.initial_scale]

    def _initialize(self):
        for weights in self.parameters[:5]:
            self.weights_init.initialize(weights, self.rng)
        self.children[1].initialize()

    @recurrent(sequences=['inputs', 'mask'],
               states=['states', 'cells', 'location', 'scale'],
               contexts=[], outputs=['states', 'cells', 'location', 'scale'])
    def apply(self, inputs, states, location, scale, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        nonlinearity = self.children[0].apply
        mlp = self.children[1]
        cropper = self.children[2]

        downn_sampled_input = cropper.apply(
            inputs.reshape((self.batch_size, 1,) + self.image_shape),
            np.array([list(self.image_shape)]),
            tensor.constant(
                (self.batch_size *
                    [[self.patch_shape[0] / 2,
                     self.patch_shape[1] / 2]])).astype('float32'),
            self.batch_size * tensor.constant(
                [[self.rescaling_factor, ] * 2]).astype('float32'))
        downn_sampled_input = downn_sampled_input.flatten(ndim=2)

        # rescaling back we we want to feed it back to MLP.
        location = (location * 2 / self.image_shape[0]) - 1
        scale = scale - self.min_scale - 1
        mlp_output = mlp.apply(tensor.concatenate(
            [downn_sampled_input, location, scale, states], axis=1))
        # To range the location between 0 and image_shape
        location = (mlp_output[:, 0:2] + 1) * self.image_shape[0] / 2
        location.name = 'location'
        # To range the scale between its min and max values
        scale = (mlp_output[:, 2:4] + 1) + self.min_scale
        scale.name = 'scale'

        patch = cropper.apply(
            inputs.reshape((self.batch_size, 1,) + self.image_shape),
            np.array([list(self.image_shape)]),
            location,
            scale)

        patch = tensor.concatenate([patch.flatten(ndim=2), location, scale],
                                   axis=1)
        transformed_patch = tensor.dot(patch, self.W_patch)

        activation = tensor.dot(states, self.W_state) + transformed_patch
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      cells * self.W_cell_to_in)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          cells * self.W_cell_to_forget)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       next_cells * self.W_cell_to_out)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells, location, scale

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0),
                tensor.repeat(self.initial_location[None, :], batch_size, 0),
                tensor.repeat(self.initial_scale[None, :], batch_size, 0)]
