import numpy
import theano
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.transformers import Transformer
floatX = theano.config.floatX


def get_mnist_streams(num_train_examples, batch_size):
    from fuel.datasets import MNIST
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


class PreprocessTransformer(Transformer):
    def __init__(self, data_stream, **kwargs):
        super(PreprocessTransformer, self).__init__(
            data_stream, **kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        normed_feat = data[0] / 255.0
        normed_feat = normed_feat.astype('float32')
        normed_feat = normed_feat[:, 5:15]
        B, T, X, Y, C = normed_feat.shape
        transformed_data.append(
            numpy.swapaxes(
                normed_feat.reshape(
                    (B, T, X * Y * C)),
                0, 1))
        # Now the data shape should be T x B x F
        transformed_data.append(data[1])
        return transformed_data


class ClutteredMNISTVideo(H5PYDataset):
    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', False)
        super(ClutteredMNISTVideo, self).__init__(
            "/data/lisatmp3/cooijmat/datasets/cluttered-mnist-video/cluttered-mnist-video.hdf5",
            which_sets, **kwargs)


def get_mnist_video_streams(batch_size):
    train_dataset = ClutteredMNISTVideo(which_sets=["train"])
    valid_dataset = ClutteredMNISTVideo(which_sets=["valid"])
    train_ind = numpy.arange(train_dataset.num_examples)
    valid_ind = numpy.arange(valid_dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(train_ind)
    rng.shuffle(valid_ind)

    train_datastream = DataStream.default_stream(
        train_dataset,
        iteration_scheme=ShuffledScheme(train_ind, batch_size))
    train_datastream = PreprocessTransformer(train_datastream)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = PreprocessTransformer(valid_datastream)

    return train_datastream, valid_datastream
