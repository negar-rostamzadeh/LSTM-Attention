# adapted from https://github.com/ballasn/LeViRe/blob/master/blocks/bricks/conv3d.py

from theano.sandbox.cuda.dnn import dnn_conv3d, dnn_pool

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, lazy
from blocks.roles import add_role, FILTER, BIAS
from blocks.utils import shared_floatx_nans

from theano.sandbox.cuda.blas import GpuCorr3dMM

class Convolutional(Initializable):
    """
    Performs a 3D convolution.

    Parameters
    ----------
    filter_size : tuple
        The duration, height and width of the filters (also called *kernels*).
    num_filters : int
        Number of filters per channel.
    num_channels : int
        Number of input channels in the video.
    batch_size : int, optional
        Number of examples per batch. If given, this will be passed to
        Theano convolution operator, possibly resulting in faster
        execution.
    input_size : tuple, optional
        The height and width of the input (video or feature map). If given,
        this will be passed to the Theano convolution operator, resulting
        in possibly faster execution times.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, for details. Defaults to 'valid'.
    shared_bias: FIXME

    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
                 input_size=None, step=(1, 1, 1), border_mode='valid',
                 cudnn_impl=False, shared_bias=False, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.input_size = input_size
        self.step = step
        self.border_mode = border_mode
        self.shared_bias = shared_bias
        self.cudnn_impl = cudnn_impl

    @property
    def padding(self):
        if self.border_mode == "valid":
            return (0, 0, 0)
        if self.border_mode == "full":
            return tuple((s - 1) / 2 for s in self.filter_size)
        else:
            return tuple(self.border_mode)

    def _allocate(self):
        W = shared_floatx_nans((self.num_filters, self.num_channels) +
                               self.filter_size, name='W')
        add_role(W, FILTER)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            if self.shared_bias:
                b = shared_floatx_nans(self.num_filters, name='b')
            else:
                b = shared_floatx_nans(self.get_dim('output'), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    def _initialize(self):
        if self.use_bias:
            W, b = self.parameters
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.parameters
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 5D tensor with the axes representing batch size, number of
            channels, height, width and time.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A 5D tensor of filtered images (feature maps) with dimensions
            representing batch size, number of filters, feature map height,
            feature map width and feature map time.
        """
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters

        if self.cudnn_impl:
            output = dnn_conv3d(input_, W,
                                subsample=tuple(self.kernel_stride),
                                border_mode=self.padding)
        else:
            output = GpuCorr3dMM(subsample=tuple(self.step),
                                 pad=self.padding)(input_, W)
        if self.use_bias:
            if self.shared_bias:
                output += b.dimshuffle('x', 0, 'x', 'x', 'x')
            else:
                output += b.dimshuffle('x', 0, 1, 2, 3)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.input_size,
        if name == 'output':
            return ((self.num_filters,) +
                    tuple(((i + 2*pad - k) // d + 1)
                          for i, k, d, pad in zip(self.input_size,
                                                  self.filter_size,
                                                  self.step,
                                                  self.padding)))
        return super(Convolutional, self).get_dim(name)


class MaxPooling(Initializable, Feedforward):
    """Max pooling layer.

    Parameters
    ----------
    pooling_size : tuple
        The height, width and time of the pooling region i.e. this is the factor
        by which your input's last two dimensions will be downscaled.
    step : tuple, optional
        The vertical, horizontal and time shift (stride) between pooling regions.
        By default this is equal to `pooling_size`. Setting this to a lower
        number results in overlapping pooling regions.
    input_dim : tuple, optional
        A tuple of integers representing the shape of the input.
    """
    @lazy(allocation=['pooling_size'])
    def __init__(self, pooling_size, step=None, num_channels=None, input_size=None, **kwargs):
        super(MaxPooling, self).__init__(**kwargs)

        self.num_channels = num_channels
        self.input_size = input_size
        self.pooling_size = pooling_size
        self.step = step

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the pooling (subsampling) transformation.
        """
        if self.pooling_size == (1, 1, 1):
            return input_
        # Pooling on last two dimensions
        input__shape = input_.shape
        input_ = input_.reshape((input__shape[0], input__shape[1] * input__shape[2], input__shape[3], input__shape[4]))
        p = dnn_pool(img=input_, ws=tuple(self.pooling_size[1:]), stride=tuple(self.step[1:]))
        p_shape = p.shape
        p = p.reshape((p_shape[0], input__shape[1], input__shape[2], p_shape[2], p_shape[3]))
        # Pooling on first dimension
        p_shape = p.shape
        p = p.reshape((p_shape[0], p_shape[1], p_shape[2], p_shape[3] * p_shape[4]))
        output = dnn_pool(img=p, ws=(self.pooling_size[0], 1), stride=(self.step[0], 1))
        output_shape = output.shape
        output = output.reshape((output_shape[0], output_shape[1], output_shape[2], p_shape[3] , p_shape[4]))
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_size
        if name == 'output':
            out_shape = ((self.num_channels,) +
                         tuple((a - b)//c + 1 for a, b, c in
                               zip(self.input_size, self.pooling_size, self.step)))
            return out_shape


class ConvolutionalActivation(Sequence, Initializable):
    """A convolution followed by an activation function.

    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply after convolution (i.e.
        the nonlinear activation function)

    See Also
    --------
    :class:`ConvolutionalActivation` : For the documentation of other parameters.

    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, activation, filter_size, num_filters, num_channels,
                 batch_size=None, input_size=None, step=(1, 1, 1),
                 border_mode='valid', **kwargs):
        self.convolution = Convolutional()

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.input_size = input_size
        self.step = step
        self.border_mode = border_mode

        super(ConvolutionalActivation, self).__init__(
            application_methods=[self.convolution.apply, activation],
            **kwargs)

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'step', 'border_mode',
                     'batch_size', 'num_channels', 'input_size']:
            setattr(self.convolution, attr, getattr(self, attr))

    def get_dim(self, name):
        # TODO The name of the activation output doesn't need to be `output`
        return self.convolution.get_dim(name)


class ConvolutionalLayer(Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.
    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling. Needed for ``__init__``.

    See Also
    --------
    :class:`Convolutional` : Documentation of convolution arguments.
    :class:`MaxPooling` : Documentation of pooling arguments.

    Notes
    -----
    Uses max pooling.

    """
    @lazy(allocation=['filter_size', 'num_filters', 'pooling_size',
                      'num_channels'])
    def __init__(self, activation, filter_size, num_filters, pooling_size,
                 num_channels, conv_step=(1, 1, 1), pooling_step=None,
                 batch_size=None, input_size=None, border_mode='valid',
                 cudnn_impl=False,
                 batch_normalize=False, pool_mode='max', **kwargs):

        self.convolution = ConvolutionalActivation(activation, batch_normalize=batch_normalize)
        self.pooling = MaxPooling()
        super(ConvolutionalLayer, self).__init__(application_methods=[self.convolution.apply,
                                                               self.pooling.apply], **kwargs)
        self.convolution.name = self.name + '_convolution'
        self.pooling.name = self.name + '_pooling'

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.pooling_size = pooling_size
        self.conv_step = conv_step
        self.pooling_step = pooling_step
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.input_size = input_size
        self.pool_mode = pool_mode

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'num_channels',
                     'batch_size', 'border_mode', 'input_size']:
            setattr(self.convolution, attr, getattr(self, attr))
        self.convolution.step = self.conv_step
        self.convolution._push_allocation_config()
        if self.input_size is not None:
            pooling_input_dim = self.convolution.get_dim('output')
        else:
            pooling_input_dim = None
        self.pooling.input_dim = pooling_input_dim
        self.pooling.pooling_size = self.pooling_size
        self.pooling.step = self.pooling_step
        self.pooling.batch_size = self.batch_size

    def get_dim(self, name):
        if name == 'input_':
            return self.convolution.get_dim('input_')
        if name == 'output':
            return self.pooling.get_dim('output')
        return super(ConvolutionalLayer, self).get_dim(name)


class ConvolutionalSequence(Sequence, Initializable, Feedforward):
    """A sequence of convolutional operations.

    Parameters
    ----------
    layers : list
        List of convolutional bricks (i.e. :class:`ConvolutionalActivation`
        or :class:`ConvolutionalLayer`)
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer.
    batch_size : int, optional
        Number of images in batch. If given, will be passed to
        theano's convolution operator resulting in possibly faster
        execution.
    input_size : tuple, optional
        Width and height of the input (image/featuremap). If given,
        will be passed to theano's convolution operator resulting in
        possibly faster execution.

    Notes
    -----
    The passed convolutional operators should be 'lazy' constructed, that
    is, without specifying the batch_size, num_channels and input_size. The
    main feature of :class:`ConvolutionalSequence` is that it will set the
    input dimensions of a layer to the output dimensions of the previous
    layer by the :meth:`~.Brick.push_allocation_config` method.

    """
    @lazy(allocation=['num_channels'])
    def __init__(self, layers, num_channels, batch_size=None, input_size=None,
                 image_size=None, **kwargs):
        self.layers = layers
        self.input_size = input_size or image_size
        self.num_channels = num_channels
        self.batch_size = batch_size

        application_methods = [brick.apply for brick in layers]
        super(ConvolutionalSequence, self).__init__(
            application_methods=application_methods, **kwargs)

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.input_size
        if name == 'output':
            return self.layers[-1].get_dim(name)
        return super(ConvolutionalSequence, self).get_dim(name)

    def _push_allocation_config(self):
        num_channels = self.num_channels
        input_size = self.input_size
        for i, layer in enumerate(self.layers):
            layer.input_size = input_size
            layer.num_channels = num_channels
            layer.batch_size = self.batch_size

            # Push input dimensions to children
            layer._push_allocation_config()

            # Retrieve output dimensions
            # and set it for next layer
            if layer.input_size is not None:
                output_shape = layer.get_dim('output')
                input_size = output_shape[1:]
                num_channels = output_shape[0]
            else:
                num_channels = layer.num_filters
