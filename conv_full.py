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
        res = expressions['outloss']
        y_hat = res.flatten(ndim=2)

        import pdb; pdb.set_trace()

        ### Build Cost
        cost = CategoricalCrossEntropy().apply(y, y_hat)
        cost = T.cast(cost, theano.config.floatX)
        cost.name = 'cross_entropy'

        y_pred = T.argmax(y_hat, axis=1)
        misclass = T.cast(T.mean(T.neq(y_pred, y)), theano.config.floatX)
        misclass.name = 'misclass'

        monitored_channels = []
        monitored_quantities = [cost, misclass, y_hat, y_pred]
        model = Model(cost)

        training_cg = ComputationGraph(monitored_quantities)
        inference_cg = ComputationGraph(monitored_quantities)

        ### Get evaluation function
        #training_eval = training_cg.get_theano_function(additional_updates=bn_updates)
        training_eval = training_cg.get_theano_function()
        #inference_eval = inference_cg.get_theano_function()


        # Dataset
        test = JpegHDF5Dataset('test',
                               #name='jpeg_data_flows.hdf5',
                               load_in_memory=True)
        #mean = np.load(os.path.join(os.environ['UCF101'], 'mean.npy'))
        import pdb; pdb.set_trace()

        ### Eval
        labels = np.zeros(test.num_video_examples)
        y_hat = np.zeros((test.num_video_examples, 101))
        labels_flip = np.zeros(test.num_video_examples)
        y_hat_flip = np.zeros((test.num_video_examples, 101))

        ### Important to shuffle list for batch normalization statistic
        #rng = np.random.RandomState()
        #examples_list = range(test.num_video_examples)
        #import pdb; pdb.set_trace()
        #rng.shuffle(examples_list)

        nb_frames=1

        for i in xrange(24):
            scheme = HDF5SeqScheme(test.video_indexes,
                                   examples=test.num_video_examples,
                                   batch_size=batch_size,
                                   f_subsample=i,
                                   nb_subsample=25,
                                   frames_per_video=nb_frames)
           #for crop in ['upleft', 'upright', 'downleft', 'downright', 'center']:
            for crop in ['center']:
                stream = JpegHDF5Transformer(
                    input_size=(240, 320), crop_size=(224, 224),
                    #input_size=(256, 342), crop_size=(224, 224),
                    crop_type=crop,
                    translate_labels = True,
                    flip='noflip', nb_frames = nb_frames,
                    data_stream=ForceFloatX(DataStream(
                            dataset=test, iteration_scheme=scheme)))
                stream_flip = JpegHDF5Transformer(
                    input_size=(240, 320), crop_size=(224, 224),
                    #input_size=(256, 342), crop_size=(224, 224),
                    crop_type=crop,
                    translate_labels = True,
                    flip='flip', nb_frames = nb_frames,
                    data_stream=ForceFloatX(DataStream(
                            dataset=test, iteration_scheme=scheme)))

                ## Do the evaluation
                epoch = stream.get_epoch_iterator()
                for j, batch in enumerate(epoch):
                    output = training_eval(batch[0], batch[1])
                    # import cv2
                    # cv2.imshow('img', batch[0][0, 0, :, :, :])
                    # cv2.waitKey(160)
                    # cv2.destroyAllWindows()
                    #import pdb; pdb.set_trace()
                    labels_flip[batch_size*j:batch_size*(j+1)] = batch[1]
                    y_hat_flip[batch_size*j:batch_size*(j+1), :] += output[2]
                preds = y_hat_flip.argmax(axis=1)
                misclass =  np.sum(labels_flip != preds) / float(len(preds))
                print i, crop, "flip Misclass:", misclass

                epoch = stream_flip.get_epoch_iterator()
                for j, batch in enumerate(epoch):
                    output = training_eval(batch[0], batch[1])
                    labels[batch_size*j:batch_size*(j+1)] = batch[1]
                    y_hat[batch_size*j:batch_size*(j+1), :] += output[2]
                preds = y_hat.argmax(axis=1)
                misclass =  np.sum(labels != preds) / float(len(preds))
                print i, crop, "noflip Misclass:", misclass

                y_merge = y_hat + y_hat_flip
                preds = y_merge.argmax(axis=1)
                misclass =  np.sum(labels != preds) / float(len(preds))
                print i, crop, "avg Misclass:", misclass


        ### Compute misclass
        y_hat += y_hat_flip
        preds = y_hat.argmax(axis=1)
        misclass =  np.sum(labels != preds) / float(len(preds))
        print "Misclass:", misclass

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

