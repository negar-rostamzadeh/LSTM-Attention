import operator
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

import numpy as np
import theano
import theano.tensor as T

from blocks.bricks.base import application, Brick
from blocks.bricks import Initializable

import bricks
import initialization

import masonry

floatX = theano.config.floatX

# this belongs on RecurrentAttentionModel as a static method, but that breaks pickling
def static_map_to_input_space(location, scale, patch_shape, image_shape):
    # linearly map locations from (-1, 1) to image index space
    location = (location + 1) / 2 * image_shape
    # disallow negative scale
    scale *= scale > 0
    # translate scale such that scale = 0 corresponds to shrinking the
    # full image to fit into the patch, and the model can only zoom in
    # beyond that.  i.e. by default the model looks at a very coarse
    # version of the image, and can choose to selectively refine
    # regions
    scale += patch_shape / image_shape
    return location, scale

class RecurrentAttentionModel(bricks.BaseRecurrent, bricks.Initializable):
    def __init__(self, hidden_dim, cropper,
                 attention_state_name, hyperparameters, **kwargs):
        super(RecurrentAttentionModel, self).__init__(**kwargs)

        self.rnn = bricks.RecurrentStack(
            [bricks.LSTM(activation=bricks.Tanh(), dim=hidden_dim),
             bricks.LSTM(activation=bricks.Tanh(), dim=hidden_dim)],
            weights_init=initialization.NormalizedInitialization(
                initialization.IsotropicGaussian()),
            biases_init=initialization.Constant(0))

        # name of the RNN state that determines the parameters of the next glimpse
        self.attention_state_name = attention_state_name

        self.cropper = cropper
        self.construct_locator(**hyperparameters)
        self.construct_merger(**hyperparameters)

        self.embedder = bricks.Linear(
            name="embedder",
            input_dim=self.response_mlp.output_dim,
            output_dim=4*self.rnn.get_dim("states"),
            use_bias=True,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(1))
        
        # don't let blocks touch my children
        self.initialization_config_pushed = True

        self.children.extend([self.rnn, self.cropper, self.embedder])

        # states aren't known until now
        self.apply.outputs = self.rnn.apply.outputs
        self.compute_initial_state.outputs = self.rnn.apply.outputs

    def construct_merger(self, n_spatial_dims, n_channels,
                         patch_shape, response_dim, patch_cnn_spec,
                         patch_mlp_spec, merge_mlp_spec,
                         response_mlp_spec, batch_normalize,
                         batch_normalize_patch, **kwargs):
        # construct patch interpretation network
        patch_transforms = []
        if patch_cnn_spec:
            patch_transforms.append(masonry.construct_cnn(
                name="patch_cnn",
                layer_specs=patch_cnn_spec,
                input_shape=patch_shape,
                n_channels=n_channels,
                batch_normalize=batch_normalize_patch))
            shape = patch_transforms[-1].get_dim("output")
        else:
            shape = (n_channels,) + tuple(patch_shape)
        patch_transforms.append(bricks.FeedforwardFlattener(input_shape=shape))
        if patch_mlp_spec:
            patch_transforms.append(masonry.construct_mlp(
                name="patch_mlp",
                hidden_dims=patch_mlp_spec,
                input_dim=patch_transforms[-1].output_dim,
                weights_init=initialization.Orthogonal(),
                biases_init=initialization.Constant(0),
                batch_normalize=batch_normalize_patch))
        self.patch_transform = bricks.FeedforwardSequence(
            [brick.apply for brick in patch_transforms], name="ffs")

        # construct theta interpretation network
        self.merge_mlp = masonry.construct_mlp(
            name="merge_mlp",
            input_dim=2*n_spatial_dims,
            hidden_dims=merge_mlp_spec,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0),
            batch_normalize=batch_normalize)

        # construct what-where merger network
        self.response_merge = bricks.Merge(
            input_names="area patch".split(),
            input_dims=[self.merge_mlp.output_dim,
                        self.patch_transform.output_dim],
            output_dim=response_dim,
            prototype=bricks.Linear(
                use_bias=False,
                weights_init=initialization.Orthogonal(),
                biases_init=initialization.Constant(0)),
            child_prefix="response_merge")
        self.response_merge_activation = bricks.NormalizedActivation(
            shape=[response_dim],
            name="response_merge_activation",
            batch_normalize=batch_normalize)

        self.response_mlp = masonry.construct_mlp(
            name="response_mlp",
            hidden_dims=response_mlp_spec,
            input_dim=response_dim,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0),
            batch_normalize=batch_normalize)

        self.children.extend([
            self.response_merge_activation,
            self.response_merge,
            self.patch_transform,
            self.merge_mlp,
            self.response_mlp])

    def construct_locator(self, locate_mlp_spec, n_spatial_dims,
                          location_std, scale_std, batch_normalize,
                          **kwargs):
        self.n_spatial_dims = n_spatial_dims

        self.locate_mlp = masonry.construct_mlp(
            name="locate_mlp",
            input_dim=self.get_dim(self.attention_state_name),
            hidden_dims=locate_mlp_spec,
            weights_init=initialization.Orthogonal(),
            biases_init=initialization.Constant(0),
            batch_normalize=batch_normalize)
        self.theta_from_area = bricks.Linear(
            input_dim=self.locate_mlp.output_dim,
            output_dim=2*n_spatial_dims,
            name="theta_from_area",
            # normalize columns because the fan-in is large
            weights_init=initialization.NormalizedInitialization(
                initialization.IsotropicGaussian()),
            # initialize location biases to zero and scale biases to one
            # so the model will zoom in by default
            biases_init=initialization.Constant(np.array(
                [0.] * n_spatial_dims + [1.] * n_spatial_dims)))

        self.T_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(12345)
        self.location_std = location_std
        self.scale_std = scale_std

        self.children.extend([
            self.locate_mlp,
            self.theta_from_area])

    def get_dim(self, name):
        try:
            return self.rnn.get_dim(name)
        except:
            return super(RecurrentAttentionModel, self).get_dim(name)

    @application
    def apply(self, x, x_shape, **states):
        location, scale = self.locate(states[self.attention_state_name])
        patch = self.crop(x, x_shape, location, scale)
        u = self.embedder.apply(self.merge(patch, location, scale))
        states = self.rnn.apply(inputs=u, iterate=False, as_dict=True, **states)
        return tuple(states.values())
        
    def locate(self, h):
        area = self.locate_mlp.apply(h)
        theta = self.theta_from_area.apply(area)
        location, scale = (theta[:, :self.n_spatial_dims],
                           theta[:, self.n_spatial_dims:])
        location += self.T_rng.normal(location.shape, std=self.location_std)
        scale += self.T_rng.normal(scale.shape, std=self.scale_std)
        return location, scale

    def merge(self, patch, location, scale):
        patch = self.patch_transform.apply(patch)
        area = self.merge_mlp.apply(T.concatenate([location, scale], axis=1))
        response = self.response_merge.apply(area, patch)
        response = self.response_merge_activation.apply(response)
        return self.response_mlp.apply(response)

    @application
    def compute_initial_state(self, x, x_shape):
        batch_size = x_shape.shape[0]
        initial_states = self.rnn.initial_states(batch_size, as_dict=True)
        # condition on initial shrink-to-fit patch
        location = T.alloc(T.cast(0.0, floatX),
                           batch_size, self.cropper.n_spatial_dims)
        scale = T.zeros_like(location)
        patch = self.crop(x, x_shape, location, scale)
        u = self.embedder.apply(self.merge(patch, location, scale))
        conditioned_states = self.rnn.apply(as_dict=True, inputs=u, iterate=False, **initial_states)
        return tuple(conditioned_states.values())

    def crop(self, x, x_shape, location, scale):
        true_location, true_scale = self.map_to_input_space(x_shape, location, scale)
        patch = self.cropper.apply(x, x_shape, true_location, true_scale)
        self.add_auxiliary_variable(location, name="location")
        self.add_auxiliary_variable(scale, name="scale")
        self.add_auxiliary_variable(true_location, name="true_location")
        self.add_auxiliary_variable(true_scale, name="true_scale")
        self.add_auxiliary_variable(patch, name="patch")
        return patch

    def map_to_input_space(self, image_shape, location, scale):
        return static_map_to_input_space(
            location, scale,
            T.cast(self.cropper.patch_shape, floatX),
            T.cast(image_shape, floatX))
