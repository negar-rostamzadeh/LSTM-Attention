import logging
import os
import time
import numpy as np
import theano.tensor as T
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Rectifier, Softmax, MLP
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams
from datasets import get_mnist_video_streams
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
# from model import LSTMModel
from LSTM_attention_model import LSTMAttention
logger = logging.getLogger('main')


def setup_model():
    # shape: T x B x F
    input_ = T.tensor3('features')
    # shape: B
    target = T.lvector('targets')
    model = LSTMAttention(input_dim=10000, dim=500,
                          mlp_hidden_dims=[2000, 500, 4],
                          batch_size=100,
                          image_shape=(100, 100),
                          patch_shape=(28, 28),
                          weights_init=IsotropicGaussian(0.01),
                          biases_init=Constant(0))
    model.initialize()
    h, c = model.apply(input_)
    classifier = MLP([Rectifier(), Softmax()], [500, 100, 10],
                     weights_init=IsotropicGaussian(0.01),
                     biases_init=Constant(0))
    classifier.initialize()

    probabilities = classifier.apply(h[-1])
    cost = CategoricalCrossEntropy().apply(target, probabilities)
    error_rate = MisclassificationRate().apply(target, probabilities)

    return cost, error_rate


def train(cost, error_rate, batch_size=100, num_epochs=150):
    # Setting Loggesetr
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/memory_' + timestr
    log_path = os.path.join(save_path, 'log.txt')
    os.makedirs(save_path)
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Training
    blocks_model = Model(cost)
    all_params = blocks_model.parameters
    print "Number of found parameters:" + str(len(all_params))
    print all_params

    training_algorithm = GradientDescent(
        cost=cost, parameters=all_params,
        step_rule=Adam(learning_rate=0.001))

    # training_algorithm = GradientDescent(
    #     cost=cost, params=all_params,
    #     step_rule=Scale(learning_rate=model.default_lr))

    monitored_variables = [cost, error_rate]

    # the rest is for validation
    # train_data_stream, valid_data_stream = get_mnist_streams(
    #     50000, batch_size)
    train_data_stream, valid_data_stream = get_mnist_video_streams(batch_size)

    train_monitoring = TrainingDataMonitoring(
        variables=monitored_variables,
        prefix="train",
        after_epoch=True)

    valid_monitoring = DataStreamMonitoring(
        variables=monitored_variables,
        data_stream=valid_data_stream,
        prefix="valid",
        after_epoch=True)

    main_loop = MainLoop(
        algorithm=training_algorithm,
        data_stream=train_data_stream,
        model=blocks_model,
        extensions=[
            train_monitoring,
            valid_monitoring,
            FinishAfter(after_n_epochs=num_epochs),
            SaveParams('valid_MSE', blocks_model, save_path),
            SaveLog(save_path, after_epoch=True),
            Printing()])
    main_loop.run()


def evaluate(model, load_path):
    with open(load_path + '/trained_params_best.npz') as f:
        loaded = np.load(f)
        blocks_model = Model(model)
        params_dicts = blocks_model.params
        params_names = params_dicts.keys()
        for param_name in params_names:
            param = params_dicts[param_name]
            assert param.get_value().shape == loaded[param_name].shape
            param.set_value(loaded[param_name])

if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        cost, error_rate = setup_model()
        train(cost, error_rate)
