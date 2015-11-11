import logging
import os
import sys
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
from utils import SaveLog, SaveParams, Glorot, visualize_attention, LRDecay
from utils import plot_curves
from datasets import (get_cmv_v2_len10_streams,
                      get_cmv_v2_len20_streams,
                      get_cmv_v2_64_len20_streams)
from blocks.initialization import Constant
from blocks.graph import ComputationGraph
from LSTM_attention_model import LSTMAttention
from blocks.monitoring import aggregation
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
floatX = theano.config.floatX
logger = logging.getLogger('main')
image_shape = (100, 100)


def setup_model():
    # shape: T x B x F
    input_ = T.tensor3('features')
    # shape: B
    target = T.lvector('targets')
    model = LSTMAttention(dim=256,
                          mlp_hidden_dims=[128, 3],
                          batch_size=100,
                          image_shape=image_shape,
                          patch_shape=(24, 24),
                          weights_init=Glorot(),
                          biases_init=Constant(0))
    model.initialize()
    h, c, location, scale, patch, downn_sampled_input = model.apply(input_)
    classifier = MLP([Rectifier(), Softmax()], [256, 128, 10],
                     weights_init=Glorot(),
                     biases_init=Constant(0))
    model.h = h
    model.location = location
    model.scale = scale
    model.patch = patch
    model.downn_sampled_input = downn_sampled_input
    classifier.initialize()

    probabilities = classifier.apply(h[-1])
    cost = CategoricalCrossEntropy().apply(target, probabilities)
    cost.name = 'CE'
    error_rate = MisclassificationRate().apply(target, probabilities)
    error_rate.name = 'ER'
    model.cost = cost

    location_5_avg = T.mean(location[5, 0])
    location_5_avg.name = 'location_5_avg'
    location_20_avg = T.mean(location[-1, 0])
    location_20_avg.name = 'location_20_avg'

    scale_0 = T.mean(scale[0, 0])
    scale_0.name = 'scale_0'
    scale_5 = T.mean(scale[5, 0])
    scale_5.name = 'scale_5'
    scale_m1 = T.mean(scale[-1, 0])
    scale_m1.name = 'scale_m1'

    monitorings = [error_rate,
                   scale_0, scale_5, scale_m1]
    model.monitorings = monitorings

    return model


def train(model, lrs, get_streams, batch_size=100, num_epochs=600):
    cost = model.cost
    monitorings = model.monitorings
    # Setting Loggesetr
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/dataset2_len10_high_lr_' + timestr
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

    default_lr = np.float32(1e-3)
    lr_var = theano.shared(default_lr, name="learning_rate")

    clipping = StepClipping(threshold=np.cast[floatX](5))

    # sgd_momentum = Momentum(
    #     learning_rate=0.0001,
    #     momentum=0.95)
    # step_rule = CompositeRule([clipping, sgd_momentum])

    adam = Adam(learning_rate=lr_var)
    step_rule = CompositeRule([clipping, adam])
    training_algorithm = GradientDescent(
        cost=cost, parameters=all_params,
        step_rule=step_rule)

    monitored_variables = [
        cost,
        aggregation.mean(training_algorithm.total_gradient_norm)] + monitorings

    blocks_model = Model(cost)

    monitored_variables.append(lr_var)

    train_data_stream, valid_data_stream = get_streams(batch_size)

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
            SaveParams('valid_ER',
                       blocks_model, save_path),
            SaveLog(save_path, after_epoch=True),
            ProgressBar(),
            LRDecay(lr_var, lrs, [80, 400],
                    after_epoch=True),
            Printing()])
    main_loop.run()


def evaluate(model, load_path, plot):
    with open(load_path + 'trained_params_best.npz') as f:
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

    if plot:
        train_data_stream, valid_data_stream = get_streams(100)
        # T x B x F
        data = train_data_stream.get_epoch_iterator().next()
        cg = ComputationGraph(model.cost)
        f = theano.function(cg.inputs, [model.location, model.scale],
                            on_unused_input='ignore',
                            allow_input_downcast=True)
        res = f(data[1], data[0])
        for i in range(10):
            visualize_attention(data[0][:, i, :],
                                res[0][:, i, :], res[1][:, i, :],
                                image_shape=image_shape, prefix=str(i))

        plot_curves(path=load_path,
                    to_be_plotted=['train_categoricalcrossentropy_apply_cost',
                                   'valid_categoricalcrossentropy_apply_cost'],
                    yaxis='Cross Entropy',
                    titles=['train', 'valid'],
                    main_title='CE')

        plot_curves(path=load_path,
                    to_be_plotted=['train_learning_rate',
                                   'train_learning_rate'],
                    yaxis='lr',
                    titles=['train', 'train'],
                    main_title='lr')

        plot_curves(path=load_path,
                    to_be_plotted=['train_total_gradient_norm',
                                   'valid_total_gradient_norm'],
                    yaxis='GradientNorm',
                    titles=['train', 'valid'],
                    main_title='GradientNorm')

        for grad in ['_total_gradient_norm',
                     '_total_gradient_norm',
                     '_/lstmattention.W_patch_grad_norm',
                     '_/lstmattention.W_state_grad_norm',
                     '_/lstmattention.initial_cells_grad_norm',
                     '_/lstmattention.initial_location_grad_norm',
                     '_/lstmattention/lstmattention_mlp/linear_0.W_grad_norm',
                     '_/lstmattention/lstmattention_mlp/linear_1.W_grad_norm',
                     '_/mlp/linear_0.W_grad_norm',
                     '_/mlp/linear_1.W_grad_norm']:
            plot_curves(path=load_path,
                        to_be_plotted=['train' + grad,
                                       'valid' + grad],
                        yaxis='GradientNorm',
                        titles=['train',
                                'valid'],
                        main_title=grad.replace(
                            "_", "").replace("/", "").replace(".", ""))

        plot_curves(path=load_path,
                    to_be_plotted=[
                        'train_misclassificationrate_apply_error_rate',
                        'valid_misclassificationrate_apply_error_rate'],
                    yaxis='Error rate',
                    titles=['train', 'valid'],
                    main_title='Error')
        print 'plot printed'


if __name__ == "__main__":
        lr = str(sys.argv[1])
        dataset = str(sys.argv[2])
        if lr == 'low':
            lrs = [1e-5, 1e-6]
        elif lr == 'med':
            lrs = [1e-4, 1e-5]
        elif lr == 'high':
            lrs = [1e-3, 1e-4]

        if dataset == 'cmv_v2_len10':
            get_streams = get_cmv_v2_len10_streams
        elif dataset == 'cmv_v2_len20':
            get_streams = get_cmv_v2_len20_streams
        elif dataset == 'cmv_v2_64_len20':
            get_streams = get_cmv_v2_64_len20_streams

        logging.basicConfig(level=logging.INFO)
        model = setup_model()
        # ds, _ = get_cmv_v2_streams(100)
        # data = ds.get_epoch_iterator(as_dict=True).next()
        # inputs = ComputationGraph(model.patch).inputs
        # f = theano.function(inputs, [model.location, model.scale,
        #                              model.patch, model.downn_sampled_input])
        # res = f(data['features'])
        # import ipdb; ipdb.set_trace()
        # location, scale, patch, downn_sampled_input = res
        # os.makedirs('res_frames/')
        # os.makedirs('res_patch/')
        # os.makedirs('res_downn_sampled_input/')
        # for i, f in enumerate(data['features']):
        #     plt.imshow(f[0].reshape(100, 100), cmap=plt.gray(),
        #                interpolation='nearest')
        #     plt.savefig('res_frames/img_' + str(i) + '.png')
        # for i, p in enumerate(patch):
        #     plt.imshow(p[0, 0], cmap=plt.gray(), interpolation='nearest')
        #     plt.savefig('res_patch/img_' + str(i) + '.png')
        # for i, d in enumerate(downn_sampled_input):
        #     plt.imshow(d[0, 0], cmap=plt.gray(), interpolation='nearest')
        #     plt.savefig('res_downn_sampled_input/img_' + str(i) + '.png')

        # for i in range(1):
        #     visualize_attention(data['features'][:, i],
        #                         (location[:, i] + 1) * image_shape[0] / 2,
        #                         scale[:, i] + 1 + 0.24 - 0.08,
        #                         image_shape=image_shape, prefix=str(i))
        # evaluate(model, 'results/dataset2_100_lenodd10_gradclip50_lr4_till5_lr5_till60_lr6_till600_2015_11_09_at_00_24/', plot=True)
        train(model, lrs, get_streams)
