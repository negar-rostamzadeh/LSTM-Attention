import math

import theano
import theano.tensor as T
import numpy as np

floatX = theano.config.floatX

from blocks.bricks import Brick, application

import util

class LocallySoftRectangularCropper(Brick):
    def __init__(self, patch_shape, kernel, hyperparameters, **kwargs):
        super(LocallySoftRectangularCropper, self).__init__(**kwargs)
        self.patch_shape = patch_shape
        self.kernel = kernel
        self.cutoff = hyperparameters["cutoff"]
        self.batched_window = hyperparameters["batched_window"]
        self.n_spatial_dims = len(patch_shape)

    def compute_crop_matrices(self, locations, scales, Is):
        Ws = []
        for axis in xrange(self.n_spatial_dims):
            n = T.cast(self.patch_shape[axis], 'float32')
            I = T.cast(Is[axis], 'float32').dimshuffle('x', 0, 'x')    # (1, hardcrop_dim, 1)
            J = T.arange(n).dimshuffle('x', 'x', 0) # (1, 1, patch_dim)

            location = locations[:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)
            scale    = scales   [:, axis].dimshuffle(0, 'x', 'x')   # (batch_size, 1, 1)

            # map patch index into image index space
            J = (J - 0.5*n) / scale + location                      # (batch_size, 1, patch_dim)

            # compute squared pairwise distances
            dx2 = (I - J)**2 # (batch_size, hardcrop_dim, patch_dim)

            Ws.append(self.kernel.density(dx2, scale))
        return Ws

    def compute_hard_windows(self, image_shape, location, scale):
        # find topleft(front) and bottomright(back) corners for each patch
        a = location - 0.5 * (T.cast(self.patch_shape, theano.config.floatX) / scale)
        b = location + 0.5 * (T.cast(self.patch_shape, theano.config.floatX) / scale)

        # grow by three patch pixels
        a -= self.kernel.k_sigma_radius(self.cutoff, scale)
        b += self.kernel.k_sigma_radius(self.cutoff, scale)

        # clip to fit inside image and have nonempty window
        a = T.clip(a, 0,     image_shape - 1)
        b = T.clip(b, a + 1, image_shape)

        if self.batched_window:
            # take the bounding box of all windows; now the slices
            # will have the same length for each sample and scan can
            # be avoided.  comes at the cost of typically selecting
            # more of the input.
            a = a.min(axis=0, keepdims=True)
            b = b.max(axis=0, keepdims=True)

        # make integer
        a = T.cast(T.floor(a), 'int16')
        b = T.cast(T.ceil(b), 'int16')

        return a, b

    @application(inputs="image image_shape location scale".split(), outputs=['patch'])
    def apply(self, image, image_shape, location, scale):
        a, b = self.compute_hard_windows(image_shape, location, scale)

        if self.batched_window:
            patch = self.apply_inner(image, location, scale, a[0], b[0])
        else:
            def map_fn(image, image_shape, a, b, location, scale):
                # apply_inner expects a batch axis
                image = T.shape_padleft(image)
                location = T.shape_padleft(location)
                scale = T.shape_padleft(scale)

                patch = self.apply_inner(image, location, scale, a, b)

                # return without batch axis
                return patch[0]

            patch, _ = theano.map(map_fn,
                                  sequences=[image, a, b, location, scale])

        savings = (1 - T.cast((b - a).prod(axis=1), floatX) / image_shape.prod(axis=1))
        self.add_auxiliary_variable(savings, name="savings")

        return patch

    def apply_inner(self, image, location, scale, a, b):
        slices = [theano.gradient.disconnected_grad(T.arange(a[i], b[i]))
                  for i in xrange(self.n_spatial_dims)]
        hardcrop = image[
            np.index_exp[:, :] +
            tuple(slice(a[i], b[i])
                  for i in range(self.n_spatial_dims))]
        matrices = self.compute_crop_matrices(location, scale, slices)
        patch = hardcrop
        for axis, matrix in enumerate(matrices):
            patch = util.batched_tensordot(patch, matrix, [[2], [1]])
        return patch

class Gaussian(object):
    def density(self, x2, scale):
        sigma = self.sigma(scale)
        volume = T.sqrt(2*math.pi)*sigma
        return T.exp(-0.5*x2/(sigma**2)) / volume

    def sigma(self, scale):
        # letting sigma vary smoothly with scale makes sense with a
        # smooth input, but the image is discretized and beyond some
        # point the kernels become so narrow that all the pixels are
        # too far away to contribute.  the filter response fades to
        # black.
        # let's not let this happen; put a lower bound on sigma.
        yuck = scale > 1.0
        scale = (1 - yuck)*scale + yuck*1.0
        sigma = 0.5 / scale
        return sigma

    def k_sigma_radius(self, k, scale):
        # this isn't correct in multiple dimensions, but it's good enough
        return k * self.sigma(scale)


if __name__ == "__main__":
    import numpy as np
    from goodfellow_svhn import NumberTask
    import matplotlib.pyplot as plt

    batch_size = 10
    task = NumberTask(batch_size=batch_size, hidden_dim=1, shrink_dataset_by=100)
    batch = task.get_stream("valid").get_epoch_iterator(as_dict=True).next()
    x_uncentered, y = task.get_variables()
    x = task.preprocess(x_uncentered)
    n_spatial_dims = 2
    image_shape = batch["features"].shape[-n_spatial_dims:]
    patch_shape = (16, 16)
    cropper = SoftRectangularCropper(n_spatial_dims=n_spatial_dims,
                                     patch_shape=patch_shape,
                                     image_shape=image_shape,
                                     kernel=Gaussian())

    scales = 1.3**np.arange(-7, 6)
    n_patches = len(scales)

    locations = (np.ones((n_patches, batch_size, 2)) * image_shape/2).astype(np.float32)
    scales = np.tile(scales[:, np.newaxis, np.newaxis], (1, batch_size, 2)).astype(np.float32)

    Tpatches = T.stack(*[cropper.apply(x_uncentered, T.constant(location), T.constant(scale))
                         for location, scale in zip(locations, scales)])

    patches = theano.function([x_uncentered], Tpatches)(batch["features"])

    m, n = batch_size, n_patches + 1

    oh_imshow = dict(interpolation="none", cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")

    print patches.shape
    for i in xrange(m):
        image = batch["features"][i, 0]
        image_ax = plt.subplot(m, n, i * n + 1)
        plt.imshow(image, shape=image_shape, axes=image_ax, **oh_imshow)
        # remove clutter
        for side in "top left bottom right".split():
            image_ax.tick_params(which="both", **{side: "off",
                                                  "label%s"%side: "off"})

        for j in xrange(1, n):
            patch = patches[j - 1, i, 0]
            location, scale = locations[j - 1, 0, 0], scales[j - 1, 0, 0]
            patch_ax = plt.subplot(m, n, i*n + j + 1)
            plt.imshow(patch, shape=patch.shape, axes=patch_ax, **oh_imshow)
            plt.title("%3.2f" % scale)
            # remove clutter
            for side in "top left bottom right".split():
                patch_ax.tick_params(which="both", **{side: "off",
                                                      "label%s"%side: "off"})
    plt.show()

