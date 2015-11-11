import numpy as np
import sys
import logging
import math
import theano.tensor as T
from sklearn_theano.feature_extraction.caffe.googlenet import create_theano_expressions as gnet_layer_exp
from sklearn_theano.feature_extraction.caffe.vgg_flows import create_theano_expressions as stream_layer_exp
from PIL import Image

from theano import config

logger = logging.getLogger(__name__)


def _is_nan(log):
    return math.isnan(log.current_row['cross_entropy'])


class ModelTrain(object):

    def __init__(self, params, parallel_flag, weight_file):
        self.params = params
        self.parallel = parallel_flag
        self.weight_file = weight_file
        self.main_loop = self._create_main_loop()

    def fit(self):
        self.main_loop.run()

    def _create_main_loop(self):
        import theano
        tensor4 = theano.tensor.TensorType(config.floatX, (False,) * 4)
        x = tensor4("images")

        # x = x[:, :, :, :, [2, 1, 0]]
        # x_shuffled = x.dimshuffle((0, 1, 4, 2, 3)) * 255
        # x_r = x_shuffled.reshape((x_shuffled.shape[0],
        #                           x_shuffled.shape[1] * x_shuffled.shape[2],
        #                           x_shuffled.shape[3], x_shuffled.shape[4]))
        # x_r = x_r - (
        #     np.array([104, 117, 123])[None, :, None, None]).astype('float32')

        with open('data.npz') as f: data = np.load(f); data=data['data']

        expressions, input_data, param = stream_layer_exp(
            inputs=('data', x), mode='rgb')
        outputs = expressions.values()
        del outputs[1]
        from blocks.graph import ComputationGraph
        # B x T x X x Y x C
        inputs = ComputationGraph(outputs).inputs
        import theano
        f = theano.function(inputs, outputs[1])
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

        img = img.astype('float32')

        res = f(data)
        keys = expressions.keys()
        del keys[1]
        for key, r in zip(keys, res):
            print key + ': ' + str(r.shape)

        print res.shape

        import ipdb; ipdb.set_trace()

        to_save = {}
        keys = ['conv_1_1_w', 'conv_1_1_b', 'conv_1_2_w', 'conv_1_2_b', 'conv_2_1_w',
                'conv_2_1_b', 'conv_2_2_w', 'conv_2_2_b', 'conv_3_1_w', 'conv_3_1_b',
                'conv_3_2_w', 'conv_3_2_b', 'conv_3_3_w', 'conv_3_3_b', 'conv_4_1_w',
                'conv_4_1_b', 'conv_4_2_w', 'conv_4_2_b', 'conv_4_3_w', 'conv_4_3_b',
                'conv_5_1_w', 'conv_5_1_b', 'conv_5_2_w', 'conv_5_2_b', 'conv_5_3_w',
                'conv_5_3_b', 'fc6_w', 'fc6_b', 'fc7_w', 'fc7_b', 'fc8-1_w', 'fc8-1_b']
        for key_1, key_2 in zip(param.keys(), keys):
            print key_1 + ': ' + str(param[key_1].get_value().shape)
            if key_2 in ['fc6_w', 'fc7_w', 'fc8-1_w']:
                to_save[key_2] = param[key_1].get_value()[:, :, 0, 0].T
            else:
                to_save[key_2] = param[key_1].get_value()
        np.savez_compressed('VGG_CNN_params', **to_save)


if __name__ == "__main__":
    basic_params = {
        'name': '/data/lisatmp3/ballasn/conv3d/UCF101_imnet/conv_fwd',
        'batch_normalize': False,
        'dropout': 0.,

        'input_size': (224, 224, 1),
        'num_channels': 2,

        # top model
        'fc_dim1': 4096,
        'fc_dim2': 1024,
        'nb_class': 101,


        'step_rule': 'AdaM',
        'step_rule_kwargs': {
            # 'max_scaling': 1e8,
        },

        'learning_rate': 1e-3,
        'batch_size': 64,
        'n_epochs': 10000}

    # make sure results are reproducable!
    np.random.seed(12345)

    if len(sys.argv) < 1:
        print sys.argv[0], "no weight_file"
        exit(1)

    # weight_file = sys.argv[1]
    weight_file = None

    model = ModelTrain(basic_params, False, weight_file)
    # model.fit()
