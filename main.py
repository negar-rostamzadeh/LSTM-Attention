import logging
import os
import time
import numpy as np
import theano.tensor as T
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams
from datasets import get_mnist_streams, get_memory_streams
from model import MLPModel, LSTMModel
logger = logging.getLogger('main')


def setup_model():
    # input_ = T.matrix('features')
    # target = T.lmatrix('targets')
    input_ = T.tensor3('features')
    target = T.tensor3('targets')
    # model = MLPModel()
    model = LSTMModel()
    model.apply(input_, target)

    return model


def train(model, batch_size=50, num_epochs=1500):
    # Setting Logger
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/memory_' + timestr
    log_path = os.path.join(save_path, 'log.txt')
    os.makedirs(save_path)
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Training
    cost = model.outputs['cost']
    blocks_model = Model(cost)
    all_params = blocks_model.parameters
    print "Number of found parameters:" + str(len(all_params))
    print all_params

    training_algorithm = GradientDescent(
        cost=cost, params=all_params,
        step_rule=Adam(learning_rate=model.default_lr))

    # training_algorithm = GradientDescent(
    #     cost=cost, params=all_params,
    #     step_rule=Scale(learning_rate=model.default_lr))

    monitored_variables = [cost]

    # the rest is for validation
    # train_data_stream, valid_data_stream = get_mnist_streams(
    #     50000, batch_size)
    train_data_stream, valid_data_stream = get_memory_streams(20, 10)

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
    load_path = None
    logging.basicConfig(level=logging.INFO)
    model = setup_model()
    if load_path is None:
        train(model)
    else:
        evaluate(model, load_path)
