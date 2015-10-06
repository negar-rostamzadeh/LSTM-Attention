import logging
import os
import time
import numpy as np
import theano
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
def toyexample(model, batch_size=50, num_epochs=1500):
    # Setting Loggesetr
    timestr = time.strftime("%Y_%m_%d_at_%H_%M")
    save_path = 'results/memory_' + timestr
    log_path = os.path.join(save_path, 'log.txt')
    os.makedirs(save_path)
    fh = logging.FileHandler(filename=log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


    cost = model.outputs['cost']
    blocks_model = Model(cost)
    all_params = blocks_model.parameters
    print "Number of found parameters:" + str(len(all_params))
    print all_params

    training_algorithm = GradientDescent(
        cost=cost, parameters=all_params,
        step_rule=Adam(learning_rate=model.default_lr))

    monitored_variables = [cost]
    train_data_stream, valid_data_stream = get_mnist_streams(
         50000, batch_size)
  
  #  train_data_stream, valid_data_stream = get_memory_streams(20, 10)
    data = valid_data_stream.get_epoch_iterator().next()
    aks = data[0][0]
   # import ipdb; ipdb.set_trace()
    aks = aks.reshape(28, 28)
    batch_size = 10
    image_shape = np.array([[28, 28]])
    from crop import LocallySoftRectangularCropper
    from crop import Gaussian
    import theano.tensor as T
    x = T.ftensor4('features')
    hyperparameters = {}
    hyperparameters["cutoff"] = 3
    hyperparameters["batched_window"] = True 
    scales = 1.3 ** np.arange(-7, 6)
    n_patches = len(scales)
    locations = (np.ones((n_patches, batch_size, 2)) *image_shape / 2).astype(np.float32)
    scales = np.tile(scales[:, np.newaxis, np.newaxis],(1, batch_size, 2)).astype(np.float32)
    location, scale = T.constant(locations[0]), T.constant(scales[0])

    cropper = LocallySoftRectangularCropper(patch_shape=(10, 10), hyperparameters=hyperparameters, kernel=Gaussian())
    import ipdb; ipdb.set_trace()
   
    patch = cropper.apply(x, image_shape, location, scale) 
   
    f = theano.function([x], patch)
    aks = data[0][0:10].reshape(10, 1, 28, 28)
    res = f(aks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(aks[0,0,:,:], cmap=plt.gray(), interpolation='nearest', vmin=    0, vmax=1)
    plt.savefig('img1.png')
    plt.imshow(res[0,0,:,:], cmap=plt.gray(), interpolation='nearest', vmin=    0, vmax=1)
    plt.savefig('img2.png')
  #  import ipdb;ipdb.set_trace()

def train(model, batch_size=50, num_epochs=1500):
    # Setting Loggesetr
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
        logging.basicConfig(level=logging.INFO)
        model = setup_model()
        toyexample(model) 
