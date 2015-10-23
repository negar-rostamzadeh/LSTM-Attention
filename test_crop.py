import theano
import theano.tensor as T
from crop import LocallySoftRectangularCropper
from crop import Gaussian
import numpy as np
from datasets import get_mnist_video_streams
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

location = T.fmatrix()
scale = T.fmatrix()
x = T.fvector()

patch_shape = (28, 28)
image_shape = (100, 100)
hyperparameters = {}
hyperparameters["cutoff"] = 3
hyperparameters["batched_window"] = True

cropper = LocallySoftRectangularCropper(
    patch_shape=patch_shape,
    hyperparameters=hyperparameters,
    kernel=Gaussian())

patch = cropper.apply(
    x.reshape((1, 1,) + image_shape),
    np.array([list(image_shape)]),
    location,
    scale)

f = theano.function([x, location, scale], patch, allow_input_downcast=True)

tds, vds = get_mnist_video_streams(1)
image = tds.get_epoch_iterator().next()[0][2, 0, :]

import ipdb; ipdb.set_trace()

location = [[60.0, 65.0]]
scale = [[1.0, 1.0]]

patch1 = f(image, location, scale)[0, 0]

plt.imshow(
    image.reshape(image_shape),
    cmap=plt.gray(),
    interpolation='nearest')
plt.savefig('img0.png')

plt.imshow(
    patch1.reshape(patch_shape),
    cmap=plt.gray(),
    interpolation='nearest')
plt.savefig('img1.png')
