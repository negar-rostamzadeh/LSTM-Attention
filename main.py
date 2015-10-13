import logging
import os
import time
import numpy as np
import theano.tensor as T
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Rectifier, Softmax, MLP
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams, Glorot
from datasets import get_mnist_video_streams
from blocks.initialization import Constant
from LSTM_attention_model import LSTMAttention
from blocks.monitoring import aggregation
logger = logging.getLogger('main')


def setup_model():
    # shape: T x B x F
    input_ = T.tensor3('features')
    # shape: B
    target = T.lvector('targets')
    model = LSTMAttention(dim=500,
                          mlp_hidden_dims=[400, 4],
                          batch_size=100,
                          image_shape=(100, 100),
                          patch_shape=(28, 28),
                          weights_init=Glorot(),
                          biases_init=Constant(0))
    model.initialize()
    h, c, location, scale = model.apply(input_)
    classifier = MLP([Rectifier(), Softmax()], [500, 100, 10],
                     weights_init=Glorot(),
                     biases_init=Constant(0))
    classifier.initialize()

    probabilities = classifier.apply(h[-1])
    cost = CategoricalCrossEntropy().apply(target, probabilities)
    error_rate = MisclassificationRate().apply(target, probabilities)

    location_x_avg = T.mean(location[:, 0], axis=0)
    location_x_avg.name = 'location_x_avg'
    location_y_avg = T.mean(location[:, 1], axis=0)
    location_y_avg.name = 'location_y_avg'
    scale_x_avg = T.mean(scale[:, 0], axis=0)
    scale_x_avg.name = 'scale_x_avg'
    scale_y_avg = T.mean(scale[:, 1], axis=0)
    scale_y_avg.name = 'scale_y_avg'

    location_x_std = T.std(location[:, 0], axis=0)
    location_x_std.name = 'location_x_std'
    location_y_std = T.std(location[:, 1], axis=0)
    location_y_std.name = 'location_y_std'
    scale_x_std = T.std(scale[:, 0], axis=0)
    scale_x_std.name = 'scale_x_std'
    scale_y_std = T.std(scale[:, 1], axis=0)
    scale_y_std.name = 'scale_y_std'

    monitorings = [error_rate,
                   location_x_avg, location_y_avg, scale_x_avg, scale_y_avg,
                   location_x_std, location_y_std, scale_x_std, scale_y_std]

    return cost, monitorings


def train(cost, monitorings, batch_size=100, num_epochs=500):
    # Setting Loggesetr
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/mnist100x100_lr_10_4_' + timestr
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
        step_rule=Adam(learning_rate=0.0001))

    monitored_variables = [cost, aggregation.mean(training_algorithm.total_gradient_norm)] + monitorings
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
            SaveParams('valid_misclassificationrate_apply_error_rate',
                       blocks_model, save_path),
            SaveLog(save_path, after_epoch=True),
            ProgressBar(),
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
        cost, monitorings = setup_model()
        train(cost, monitorings)
