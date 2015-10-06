import os
import logging

import numpy as np

import theano
import theano.tensor as T

from blocks.filter import VariableFilter

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel import transformers

import emitters

logger = logging.getLogger(__name__)

# have the same sources for all tasks
CANONICAL_SOURCES = tuple("features shapes targets".split())

class Canonicalize(transformers.Transformer):
    produces_examples = False

    def __init__(self, stream, mapping, **kwargs):
        super(Canonicalize, self).__init__(stream, **kwargs)
        self.mapping = mapping

    @property
    def sources(self):
        return CANONICAL_SOURCES

    def transform_batch(self, batch):
        return self.mapping(batch)

class Classification(object):
    def __init__(self, batch_size, hidden_dim, shrink_dataset_by=1, **kwargs):
        self.shrink_dataset_by = shrink_dataset_by
        self.batch_size = batch_size
        self.datasets = self.load_datasets()

    def load_datasets(self):
        raise NotImplementedError()
        
    def apply_default_transformers(self, stream):
        return stream

    def get_stream_num_examples(self, which_set, monitor):
        return (self.datasets[which_set].num_examples
                / self.shrink_dataset_by)

    def get_stream(self, which_set, shuffle=True, monitor=False, num_examples=None, center=True):
        scheme_klass = ShuffledScheme if shuffle else SequentialScheme
        if num_examples is None:
            num_examples = self.get_stream_num_examples(which_set, monitor=monitor)
        scheme = scheme_klass(num_examples, self.batch_size)
        stream = DataStream.default_stream(
            dataset=self.datasets[which_set],
            iteration_scheme=scheme)
        stream = self.apply_default_transformers(stream)
        stream = Canonicalize(stream, mapping=self.preprocess)
        if center:
            stream = transformers.Mapping(stream, mapping=self.center)
        return stream

    def get_variables(self):
        variables = []
        test_batch = self.get_stream("valid").get_epoch_iterator(as_dict=True).next()
        for key in CANONICAL_SOURCES:
            value = test_batch[key]
            variable = T.TensorType(
                broadcastable=[False]*value.ndim,
                dtype=value.dtype)(key)
            variable.tag.test_value = value[:11]
            variables.append(variable)
        return variables

    def get_emitter(self, input_dim, batch_normalize, **kwargs):
        return emitters.SingleSoftmax(input_dim, self.n_classes,
                                      batch_normalize=batch_normalize)

    def monitor_channels(self, graph):
        return [VariableFilter(name=name)(graph.auxiliary_variables)[0]
                for name in "cross_entropy error_rate".split()]

    def plot_channels(self):
        return [["%s_%s" % (which_set, name) for which_set in self.datasets.keys()]
                for name in "cross_entropy error_rate".split()]
        
    def center(self, data):
        x, x_shape, y = data
        mean = self.get_mean()
        masks = np.zeros_like(x)
        for i, shape in enumerate(x_shape):
            masks[np.index_exp[i, :] + tuple(map(slice, shape))] = 1
        x_centered = x - masks * mean
        return x_centered, x_shape, y

    def get_mean(self):
        cache_dir = os.environ["PREPROCESS_CACHE"]
        try:
            os.mkdir(cache_dir)
        except OSError:
            # directory already exists. surely the end of the world.
            pass
        cache = os.path.join(cache_dir, "%s.npz" % self.name)
        try:
            data = np.load(cache)
            mean = data["mean"]
        except IOError:
            print "taking mean"
            mean = self.compute_mean()
            print "mean taken"
            try:
                np.savez(cache, mean=mean)
            except IOError, e:
                logger.error("couldn't save preprocessing cache: %s" % e)
                import ipdb; ipdb.set_trace()
        return mean

    def compute_mean(self):
        mean = 0
        n = 0
        for batch in (self.get_stream("train", center=False)
                      .get_epoch_iterator(as_dict=True)):
            x, x_shape = batch["features"], batch["shapes"]
            k = x.shape[0]
            mean = (n/float(n+k) * mean +
                    k/float(n+k) * self.compute_batch_mean(x, x_shape))
            n += k
        mean = mean.astype(np.float32)
        return mean

    def compute_batch_mean(self, x, x_shape):
        return x.mean(axis=0, keepdims=True)
