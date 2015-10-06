import logging
logger = logging.getLogger(__name__)

import itertools as it
import numbers
from theano.compile import ViewOp
from collections import OrderedDict
from blocks.utils import named_copy
from blocks.initialization import NdarrayInitialization

import theano.tensor as T

def broadcast_index(index, axes, ndim):
    dimshuffle_args = ['x'] * ndim
    if isinstance(axes, numbers.Integral):
        axes = [axes]
    for i, axis in enumerate(axes):
        dimshuffle_args[axis] = i
    return index.dimshuffle(*dimshuffle_args)

def broadcast_indices(index_specs, ndim):
    indices = []
    for index, axes in index_specs:
        indices.append(broadcast_index(index, axes, ndim))
    return indices

def subtensor(x, index_specs):
    indices = broadcast_indices(index_specs, x.ndim)
    return x[tuple(indices)]

class WithDifferentiableApproximation(ViewOp):
    __props__ = ()

    def make_node(self, fprop_output, bprop_output):
        # avoid theano wasting time computing the gradient of fprop_output
        fprop_output = theano.gradient.disconnected_grad(fprop_output)
        return gof.Apply(self, [fprop_output, bprop_output], [f.type()])

    def grad(self, wrt, input_gradients):
        import pdb; pdb.set_trace()
        # check that we need input_gradients[1] rather than input_gradients[:][1]
        return input_gradients[1]

def with_differentiable_approximation(fprop_output, bprop_output):
    return WithDifferentiableApproximation()(fprop_output, bprop_output)

# to handle non-unique monitoring channels without crashing and
# without silent loss of information
class Channels(object):
    def __init__(self):
        self.dikt = OrderedDict()

    def append(self, quantity, name=None):
        if name is not None:
            quantity = named_copy(quantity, name)
        self.dikt.setdefault(quantity.name, []).append(quantity)

    def extend(self, quantities):
        for quantity in quantities:
            self.append(quantity)

    def get_channels(self):
        channels = []
        for _, quantities in self.dikt.items():
            if len(quantities) == 1:
                channels.append(quantities[0])
            else:
                # name not unique; uniquefy
                for i, quantity in enumerate(quantities):
                    channels.append(named_copy(
                        quantity, "%s[%i]" % (quantity.name, i)))
        return channels

def dict_merge(*dikts):
    result = OrderedDict()
    for dikt in dikts:
        result.update(dikt)
    return result

def named(x, name):
    x.name = name
    return x

# from http://stackoverflow.com/a/16571630
from cStringIO import StringIO
import sys

class StdoutLines(list):
    def __enter__(self):
        self._stringio = StringIO()
        self._stdout = sys.stdout
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

import theano.tensor.basic
import theano.sandbox.cuda.blas

def batched_tensordot(a, b, axes=2):
    return theano.tensor.basic._tensordot_as_dot(
        a, b, axes,
        dot=theano.sandbox.cuda.blas.batched_dot,
        batched=True)

import theano.printing
from blocks.filter import VariableFilter
import numpy as np

def get_recurrent_auxiliaries(names, graph, n_steps=None):
    variables = []
    for name in names:
        steps = VariableFilter(name=name)(graph.auxiliary_variables)

        if n_steps is not None:
            assert len(steps) == n_steps

        # a super crude sanity check to ensure these auxiliaries are
        # actually in chronological order
        assert all(_a < _b for _a, _b in 
                   (lambda _xs: zip(_xs, _xs[1:]))
                   ([len(theano.printing.debugprint(step, file="str"))
                     for step in steps]))

        variable = T.stack(*steps)
        # move batch axis before rnn time axis
        variable = variable.dimshuffle(1, 0, *range(2, variable.ndim))
        variables.append(variable)
    return variables

from blocks.bricks.base import Brick, ApplicationCall

# attempt to fully qualify an annotated variable
def get_path(x):
    if isinstance(x, (T.TensorVariable,
                      # zzzzzzzzzzzzzzzzzzzzzzzzzzz
                      T.sharedvar.TensorSharedVariable,
                      T.compile.sharedvalue.SharedVariable)):
        paths = list(set(map(get_path, x.tag.annotations)))
        name = getattr(x.tag, "name", x.name)
        if len(paths) > 1:
            logger.warning(
                "get_path: variable %s has multiple possible origins, using first of [%s]"
                % (name, " ".join(paths)))
        return paths[0] + "/" + name
    elif isinstance(x, Brick):
        if x.parents:
            paths = list(set(map(get_path, x.parents)))
            if len(paths) > 1:
                logger.warning(
                    "get_path: brick %s has multiple parents, using first of [%s]"
                    % (x.name, " ".join(paths)))
            return paths[0] + "/" + x.name
        else:
            return "/" + x.name
    elif isinstance(x, ApplicationCall):
        return get_path(x.application.brick)
    else:
        raise TypeError()
