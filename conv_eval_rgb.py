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
        tensor5 = T.TensorType(config.floatX, (False,) * 5)
        x = tensor5("images")

        x = x[:, :, :, :, [2, 1, 0]]
        x_shuffled = x.dimshuffle((0, 1, 4, 2, 3)) * 255
        x_r = x_shuffled.reshape((x_shuffled.shape[0],
                                  x_shuffled.shape[1] * x_shuffled.shape[2],
                                  x_shuffled.shape[3], x_shuffled.shape[4]))
        x_r = x_r - (
            np.array([104, 117, 123])[None, :, None, None]).astype('float32')

        expressions, input_data, param = stream_layer_exp(
            inputs=('data', x_r), mode='rgb')
        outputs = expressions['fc8-1']
        from blocks.graph import ComputationGraph
        # B x T x X x Y x C
        inputs = ComputationGraph(outputs).inputs
        import theano
        f = theano.function(inputs, outputs)
        img = np.array(Image.open('img_4.jpeg'))
        img = img[:224, :224]
        img = img[np.newaxis, np.newaxis, :, :, :]
        img = img / 255.0
        img = img.astype('float32')

        res = f(img)
        print np.argmax(res[0, :, 0, 0])

        import ipdb; ipdb.set_trace()


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
