import unittest

import numpy as np
import theano
import theano.tensor as T

import util

class TestSubtensor(unittest.TestCase):
    def test_subtensor(self):
        ndim = 5
        index = T.constant(np.arange(6).reshape(2, 3))
        broadcasted_index = util.broadcast_index(index, (3, 2), ndim)
        self.assertEqual((1, 1, 3, 2, 1),
                         tuple(theano.function([], broadcasted_index.shape)()))

if __name__ == "__main__":
    unittest.main()

