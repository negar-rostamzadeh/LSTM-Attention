# Credits to Kyle Kastner
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.dnn import dnn_conv
from PIL import Image


def shared_normal(rng, shape, mean=0.0, stdev=0.25, name=None):
    v = np.asarray(mean + rng.standard_normal(shape) * stdev,
                   dtype=theano.config.floatX)
    return theano.shared(v, name=name)


def shared_zeros(shape, name=None):
    v = np.zeros(shape, dtype=theano.config.floatX)
    return theano.shared(v, name=name)


def _relu(x):
    return x * (x > 1E-6)


def conv_layer(input_variable, filter_shape, pool_shape,
               rng=np.random.RandomState(1999),
               layer_name='', border_mode=(1, 1)):
    filters = shared_normal(rng, filter_shape, name=layer_name + '_w')
    biases = shared_zeros(filter_shape[0], name=layer_name + '_b')
    params = [filters, biases]
    conv = (dnn_conv(input_variable, filters, border_mode=border_mode) +
            biases.dimshuffle('x', 0, 'x', 'x'))
    out = _relu(conv)

    if pool_shape is not None:
        out = max_pool_2d(out, pool_shape, ignore_border=True)
    return out, params


def fc_layer(input_variable, layer_shape,
             rng=np.random.RandomState(1999),
             layer_name=''):
    w = shared_normal(rng, layer_shape, name=layer_name + '_w')
    b = shared_zeros(layer_shape[1], name=layer_name + '_b')
    params = [w, b]
    out = _relu(T.dot(input_variable, w) + b)
    print "WARNING: ReLU is used."
    return out, params


# input: B x X x Y x C
# output: B x X x Y x C
def pre_process(x):
    xx = x.dimshuffle(0, 'x', 1, 2, 3)
    xx = xx[:, :, :, :, [2, 1, 0]]
    xx = xx.dimshuffle((0, 1, 4, 2, 3)) * 255
    xx = xx.reshape((xx.shape[0], xx.shape[1] * xx.shape[2],
                     xx.shape[3], xx.shape[4]))
    xx = xx - (
        np.array([104, 117, 123])[None, :, None, None]).astype('float32')
    return xx


def apply_convnet(x):
    # x must be : B x X x Y x C
    # output_channels, input_channels, kernel_width, kernel_height
    filters_and_pools = [['conv_1_1', (64, 3, 3, 3), None],
                         ['conv_1_2', (64, 64, 3, 3), (2, 2)],
                         ['conv_2_1', (128, 64, 3, 3), None],
                         ['conv_2_2', (128, 128, 3, 3), (2, 2)],
                         ['conv_3_1', (256, 128, 3, 3), None],
                         ['conv_3_2', (256, 256, 3, 3), None],
                         ['conv_3_3', (256, 256, 3, 3), (2, 2)],
                         ['conv_4_1', (512, 256, 3, 3), None],
                         ['conv_4_2', (512, 512, 3, 3), None],
                         ['conv_4_3', (512, 512, 3, 3), (2, 2)],
                         ['conv_5_1', (512, 512, 3, 3), None],
                         ['conv_5_2', (512, 512, 3, 3), None],
                         ['conv_5_3', (512, 512, 3, 3), (2, 2)]]
    fully_connected_layers = [['fc6', (25088, 4096)],
                              ['fc7', (4096, 4096)],
                              ['fc8-1', (4096, 101)]]
    xx = pre_process(x)
    params = []
    outputs = {'x': x}
    out = xx
    for name, filter_shape, pool_shape in filters_and_pools:
        out, l_params = conv_layer(out, filter_shape, pool_shape,
                                   layer_name=name)
        params += l_params
        outputs[name] = out
    # B x F
    out = out.flatten(2)
    for name, fc_shape in fully_connected_layers:
        out, l_params = fc_layer(out, layer_shape=fc_shape, layer_name=name)
        params += l_params
        outputs[name] = out

    with open('VGG_CNN_params.npz') as f:
        loaded = np.load(f)
        for param in params:
            assert param.get_value().shape == loaded[param.name].shape
            param.set_value(loaded[param.name])
    return outputs


img1 = np.array(Image.open('img_4.jpeg'))
img1 = img1[:224, :224]
img1 = img1[np.newaxis, np.newaxis, :, :, :]
img1 = img1 / 255.0
img1 = img1.astype('float32')

img2 = np.array(Image.open('img_3.jpg'))
img2 = img2[:224, :224]
img2 = img2[np.newaxis, np.newaxis, :, :, :]
img2 = img2 / 255.0
img2 = img2.astype('float32')

B, Time, X, Y, C = img1.shape
img = np.zeros((B + 1, Time, X, Y, C))
img[0] = img1
img[1] = img2
img = img[:, 0]
img = img.astype('float32')
x = T.tensor4('x')
outputs = apply_convnet(x)

# with open('data.npz') as f: data = np.load(f); data=data['data']
# f = theano.function([x], xx)
# f(img)
f = theano.function([x], outputs['fc8-1'])
res = f(img)
print np.argmax(res, axis=1)
import ipdb; ipdb.set_trace()
