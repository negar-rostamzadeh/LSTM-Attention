import numpy
import theano
from fuel.streams import DataStream
from fuel.transformers import Flatten
floatX = theano.config.floatX


def get_mnist_streams(num_train_examples, batch_size):
    from fuel.datasets import MNIST
    from fuel.schemes import ShuffledScheme
    dataset = MNIST(("train",))
    all_ind = numpy.arange(dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(all_ind)

    indices_train = all_ind[:num_train_examples]
    indices_valid = all_ind[num_train_examples:]

    tarin_stream = Flatten(DataStream.default_stream(
        dataset,
        iteration_scheme=ShuffledScheme(indices_train, batch_size)),
        which_sources=('features',))

    valid_stream = Flatten(DataStream.default_stream(
        dataset,
        iteration_scheme=ShuffledScheme(indices_valid, batch_size)),
        which_sources=('features',))

    return tarin_stream, valid_stream


def get_memory_streams(num_train_examples, batch_size, time_length=15, dim=2):
    from fuel.datasets import IterableDataset
    numpy.random.seed(0)
    num_sequences = num_train_examples / batch_size

    # generating random sequences
    seq_u = numpy.random.randn(num_sequences, time_length, batch_size, dim)
    seq_y = numpy.zeros((num_sequences, time_length, batch_size, dim))

    seq_y[:, 1:, :, 0] = seq_u[:, :-1, :, 0]  # 1 time-step delay
    seq_y[:, 3:, :, 1] = seq_u[:, :-3, :, 1]  # 3 time-step delay

    seq_y += 0.01 * numpy.random.standard_normal(seq_y.shape)

    dataset = IterableDataset({'features': seq_u.astype(floatX),
                               'targets': seq_y.astype(floatX)})
    tarin_stream = DataStream(dataset)
    valid_stream = DataStream(dataset)

    return tarin_stream, valid_stream
