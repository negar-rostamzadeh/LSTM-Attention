import logging
import os
import time
import numpy as np
import theano.tensor as T
import theano
from blocks.algorithms import (GradientDescent, Adam,
                               CompositeRule, StepClipping)
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Rectifier, Softmax, MLP
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams, Glorot, visualize_attention
from datasets import get_mnist_video_streams
from blocks.initialization import Constant
from blocks.graph import ComputationGraph
from LSTM_attention_model import LSTMAttention
from blocks.monitoring import aggregation
floatX = theano.config.floatX
logger = logging.getLogger('main')


def setup_model():
    # shape: T x B x F
    input_ = T.tensor3('features')
    # shape: B
    target = T.lvector('targets')
    model = LSTMAttention(dim=256,
                          mlp_hidden_dims=[256, 4],
                          batch_size=100,
                          image_shape=(100, 100),
                          patch_shape=(16, 16),
                          weights_init=Glorot(),
                          biases_init=Constant(0))
    model.initialize()
    h, c, location, scale = model.apply(input_)
    classifier = MLP([Rectifier(), Softmax()], [256 * 2, 200, 10],
                     weights_init=Glorot(),
                     biases_init=Constant(0))
    model.h = h
    model.c = c
    model.location = location
    model.scale = scale
    classifier.initialize()

    probabilities = classifier.apply(T.concatenate([h[-1], c[-1]], axis=1))
    cost = CategoricalCrossEntropy().apply(target, probabilities)
    error_rate = MisclassificationRate().apply(target, probabilities)
    model.cost = cost

    location_x_avg = T.mean(location[:, 0])
    location_x_avg.name = 'location_x_avg'
    location_y_avg = T.mean(location[:, 1])
    location_y_avg.name = 'location_y_avg'
    scale_x_avg = T.mean(scale[:, 0])
    scale_x_avg.name = 'scale_x_avg'
    scale_y_avg = T.mean(scale[:, 1])
    scale_y_avg.name = 'scale_y_avg'

    location_x_std = T.std(location[:, 0])
    location_x_std.name = 'location_x_std'
    location_y_std = T.std(location[:, 1])
    location_y_std.name = 'location_y_std'
    scale_x_std = T.std(scale[:, 0])
    scale_x_std.name = 'scale_x_std'
    scale_y_std = T.std(scale[:, 1])
    scale_y_std.name = 'scale_y_std'

    monitorings = [error_rate,
                   location_x_avg, location_y_avg, scale_x_avg, scale_y_avg,
                   location_x_std, location_y_std, scale_x_std, scale_y_std]
    model.monitorings = monitorings

    return model


def train(model, batch_size=100, num_epochs=1000):
    cost = model.cost
    monitorings = model.monitorings
    # Setting Loggesetr
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/test_' + timestr
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

    clipping = StepClipping(threshold=np.cast[floatX](10))

    adam = Adam(learning_rate=0.00001)
    step_rule = CompositeRule([adam, clipping])
    training_algorithm = GradientDescent(
        cost=cost, parameters=all_params,
        step_rule=step_rule)

    monitored_variables = [
        cost,
        aggregation.mean(training_algorithm.total_gradient_norm)] + monitorings

    blocks_model = Model(cost)
    params_dicts = blocks_model.get_parameter_dict()
    for name, param in params_dicts.iteritems():
        to_monitor = training_algorithm.gradients[param].norm(2)
        to_monitor.name = name + "_grad_norm"
        monitored_variables.append(to_monitor)
        to_monitor = param.norm(2)
        to_monitor.name = name + "_norm"
        monitored_variables.append(to_monitor)

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
        blocks_model = Model(model.cost)
        params_dicts = blocks_model.get_parameter_dict()
        params_names = params_dicts.keys()
        for param_name in params_names:
                    param = params_dicts[param_name]
                    # '/f_6_.W' --> 'f_6_.W'
                    slash_index = param_name.find('/')
                    param_name = param_name[slash_index + 1:]
                    assert param.get_value().shape == loaded[param_name].shape
                    param.set_value(loaded[param_name])

    train_data_stream, valid_data_stream = get_mnist_video_streams(100)
    # T x B x F
    data = train_data_stream.get_epoch_iterator().next()
    cg = ComputationGraph(model.cost)
    f = theano.function(cg.inputs, [model.location, model.scale],
                        on_unused_input='ignore',
                        allow_input_downcast=True)
    res = f(data[1], data[0])
    for i in range(10):
        visualize_attention(data[0][:, i, :],
                            res[0][:, i, :], res[1][:, i, :], prefix=str(i))

if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        model = setup_model()
        # evaluate(model, 'results/test_cont_adam_lr_m5_2015_10_18_at_15_35')
        train(model)
