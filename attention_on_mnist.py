import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Tanh, Softmax, WEIGHT
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
import crop
from crop import SoftRectangularCropper
from crop import Gaussian
try:
    from blocks.extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


def main(save_to, num_epochs):
    mlp = MLP([Tanh(), Softmax()], [784, 100, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    #attention ---> 
    patch_shape = (16, 16);
    image_shape = (784,100);
    import numpy
    import theano.tensor as T
    n_spatial_dims = 2
    cropper = SoftRectangularCropper(n_spatial_dims=n_spatial_dims,
                                     patch_shape=patch_shape,
                                     image_shape=image_shape,
                                     kernel=Gaussian())


    batch_size = 10    
    scales = 1.3**numpy.arange(-7, 6)
    n_patches = len(scales)
    locations = (numpy.ones((n_patches, batch_size, 2)) * image_shape/2).astype(numpy.float32)
    scales = numpy.tile(scales[:, numpy.newaxis, numpy.newaxis], (1, batch_size, 2)).astype(numpy.float32)
    Tpatches = T.stack(*[cropper.apply(x, T.constant(location), T.constant(scale))[0]
	    for location, scale in zip(locations, scales)])
    patches = theano.function([x], Tpatches)(batch['features'])

    import ipdb as pdb; pdb.set_trace()
    probs = mlp.apply(tensor.flatten(patches, outdim=2))
    cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
    error_rate = MisclassificationRate().apply(y.flatten(), probs)

    cg = ComputationGraph([cost])
    W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + .00005 * (W1 ** 2).sum() + .00005 * (W2 ** 2).sum()
    cost.name = 'final_cost'

    mnist_train = MNIST(("train",))
    mnist_test = MNIST(("test",))

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=0.1))
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      Flatten(
                          DataStream.default_stream(
                              mnist_test,
                              iteration_scheme=SequentialScheme(
                                  mnist_test.num_examples, 500)),
                          which_sources=('features',)),
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    if BLOCKS_EXTRAS_AVAILABLE:
        extensions.append(Plot(
            'MNIST example',
            channels=[
                ['test_final_cost',
                 'test_misclassificationrate_apply_error_rate'],
                ['train_total_gradient_norm']]))

    main_loop = MainLoop(
        algorithm,
        Flatten(
            DataStream.default_stream(
                mnist_train,
                iteration_scheme=SequentialScheme(
                    mnist_train.num_examples, 50)),
            which_sources=('features',)),
        model=Model(cost),
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)

