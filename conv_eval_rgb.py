import numpy as np
import sys
import os
import cPickle
import logging
import theano
import argparse
import math
import os
import subprocess
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.sandbox.cuda.basic_ops import gpu_contiguous

from blocks.bricks.conv import (Convolutional, ConvolutionalLayer, MaxPooling,
                                ConvolutionalActivation, ConvolutionalSequence)
from blocks.initialization import Constant, Uniform
from blocks.bricks import Linear, Rectifier, Softmax, Identity
from blocks.algorithms import GradientDescent, RMSProp, Scale, Momentum
from LeViRe.blocks.algorithms import AdaM
from blocks.filter import VariableFilter
from blocks.roles import PARAMETER, OUTPUT, INPUT, DROPOUT
from blocks.graph import ComputationGraph, apply_dropout
from blocks.extensions.monitoring import DataStreamMonitoring

from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector
from blocks.extensions.monitoring import TrainingDataMonitoring
#from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint
from blocks.theano_expressions import l2_norm
#from blocks.scripts import continue_training
from blocks.bricks.cost import CategoricalCrossEntropy
from theano import function

from fuel.streams import DataStream
from fuel.streams import ServerDataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Mapping, ForceFloatX


from picklable_itertools.extras import equizip
from collections import OrderedDict

from LeViRe.fuel.datasets.hdf5jpeg import JpegHDF5Dataset
#from LeViRe.fuel.transformers.googlenet import GoogleNetTransformer
from LeViRe.fuel.transformers.flows_transformer import FlowHDF5Transformer
from LeViRe.fuel.transformers.frames_transformer import JpegHDF5Transformer
from LeViRe.fuel.utils import server
from LeViRe.fuel.scheme import HDF5ShuffledScheme, HDF5SeqScheme
from LeViRe.blocks.extensions.saveload import BestValidationDumpParams
from LeViRe.blocks.bricks import MLP, TensorMLP

from LeViRe.blocks.bricks.conv_lstm import ConvLSTMLayer, ConvGRULayer, ConvLSTMSequence, ConvGRUStack, ConvGRUStackBN
from sklearn_theano.feature_extraction.caffe.googlenet import create_theano_expressions as gnet_layer_exp
from sklearn_theano.feature_extraction.caffe.vgg_flows import create_theano_expressions as stream_layer_exp

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
        # hyper parameters
        hp = self.params
        batch_size = hp['batch_size']
        biases_init = Constant(0)
        batch_normalize = hp['batch_normalize']

        ### Build fprop
        tensor5 = T.TensorType(config.floatX, (False,)*5)
        X = tensor5("images")
        #X = T.tensor4("images")
        y = T.lvector('targets')

        gnet_params = OrderedDict()
        #X_shuffled = X[:, :, :, :, [2, 1, 0]]
        #X_shuffled = gpu_contiguous(X.dimshuffle(0, 1, 4, 2, 3)) * 255

        X = X[:, :, :, :, [2, 1, 0]]
        X_shuffled = X.dimshuffle((0, 1, 4, 2, 3)) * 255
        X_r = X_shuffled.reshape((X_shuffled.shape[0],
                                  X_shuffled.shape[1]*X_shuffled.shape[2],
                                  X_shuffled.shape[3], X_shuffled.shape[4]))
        X_r = X_r - (np.array([104, 117, 123])[None, :, None, None]).astype('float32')


        expressions, input_data, param = stream_layer_exp(inputs = ('data', X_r),
                                                          mode='rgb')

       # f=theano.function(inputs=[('data', X_r), mode='rgb'],outputs= stream_layer_exp(inputs))
       # return f
	import ipdb; ipdb.set_trace()
        # Dataset
        #mean = np.load(os.path.join(os.environ['UCF101'], 'mean.npy'))

        ### Eval
        X = load_sample_image("sloth_closeup.jpg")
        ### Important to shuffle list for batch normalization statistic
        rng = np.random.RandomState()
        examples_list = range(test.num_video_examples)
        import ipdb; ipdb.set_trace()
        rng.shuffle(examples_list)

        import cv2
        cv2.imshow('X', batch[0][0, 0, :, :, :])
        cv2.waitKey(160)
        cv2.destroyAllWindows()
        import ipdb; ipdb.set_trace()


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
    'n_epochs': 10000,
}

if __name__ == "__main__":
    # make sure results are reproducable!
    np.random.seed(12345)

    if len(sys.argv) < 1:
        print sys.argv[0], "no weight_file"
        exit(1)

    #weight_file = sys.argv[1]
    weight_file = None

    model = ModelTrain(basic_params, False, weight_file)
    #model.fit()
    import ipdb as pdb; pdb.set_trace()

