import logging
import numpy as np
import theano
import theano.tensor as T
from blocks.extensions import SimpleExtension
from blocks.roles import add_role
from blocks.roles import AuxiliaryRole
from crop import LocallySoftRectangularCropper
from crop import Gaussian
from blocks.initialization import NdarrayInitialization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logger = logging.getLogger('main.utils')


class BnParamRole(AuxiliaryRole):
    pass
BNPARAM = BnParamRole()


def shared_param(init, name, cast_float32, role, **kwargs):
    if cast_float32:
        v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class LRDecay(SimpleExtension):
    def __init__(self, lr_var, lrs, until_which_epoch, **kwargs):
        super(LRDecay, self).__init__(**kwargs)
        # assert self.num_epochs == until_which_epoch[-1]
        self.iter = 0
        self.lrs = lrs
        self.until_which_epoch = until_which_epoch
        self.lr_var = lr_var

    def do(self, which_callback, *args):
        self.iter += 1
        if self.iter < self.until_which_epoch[-1]:
            lr_index = [self.iter < epoch for epoch
                        in self.until_which_epoch].index(True)
        else:
            print "WARNING: the smallest learning rate is using."
            lr_index = -1
        self.lr_var.set_value(np.float32(self.lrs[lr_index]))


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, early_stop_var, model, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.early_stop_var = early_stop_var
        self.save_path = save_path
        params_dicts = model.get_parameter_dict()
        self.params_names = params_dicts.keys()
        self.params_values = params_dicts.values()
        self.to_save = {}
        self.best_value = None
        self.add_condition(('after_training',), self.save)
        self.add_condition(('on_interrupt',), self.save)
        self.add_condition(('after_epoch',), self.do)

    def save(self, which_callback, *args):
        to_save = {}
        for p_name, p_value in zip(self.params_names, self.params_values):
            to_save[p_name] = p_value.get_value()
        path = self.save_path + '/trained_params'
        np.savez_compressed(path, **to_save)

    def do(self, which_callback, *args):
        val = self.main_loop.log.current_row[self.early_stop_var]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
            to_save = {}
            for p_name, p_value in zip(self.params_names, self.params_values):
                to_save[p_name] = p_value.get_value()
            path = self.save_path + '/trained_params_best'
            np.savez_compressed(path, **to_save)


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


def apply_act(input, act_name):
    if input is None:
        return input
    act = {
        'relu': lambda x: T.maximum(0, x),
        'leakyrelu': lambda x: T.switch(x > 0., x, 0.1 * x),
        'linear': lambda x: x,
        'softplus': lambda x: T.log(1. + T.exp(x)),
        'sigmoid': lambda x: T.nnet.sigmoid(x),
        'softmax': lambda x: T.nnet.softmax(x),
    }.get(act_name)
    if act_name == 'softmax':
        input = input.flatten(2)
    return act(input)


class Glorot(NdarrayInitialization):
    def generate(self, rng, shape):
        input_size, output_size = shape
        high = np.sqrt(6) / np.sqrt(input_size + output_size)
        m = rng.uniform(-high, high, size=shape)
        if shape == (256, 1024):
            # import ipdb; ipdb.set_trace()
            high = np.sqrt(6) / np.sqrt(256 + 1024)
            mi = rng.uniform(-high, high, size=(256, 256))
            mf = rng.uniform(-high, high, size=(256, 256))
            mc = np.identity(256) * 0.99
            mo = rng.uniform(-high, high, size=(256, 256))
            m = np.hstack([mi, mf, mc, mo])
        return m.astype(theano.config.floatX)


def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


# im: size 10000
# im_2: size 100 x 100 x 3
def show_patch_on_frame(im, location_, scale_,
                        image_shape=(100, 100), patch_shape=(16, 16)):
    img = to_rgb1(im.reshape(image_shape))
    im_2 = img + np.zeros(img.shape)
    x_0 = -(patch_shape[0] / 2) / scale_[0] + location_[0]
    x_1 = (patch_shape[0] / 2) / scale_[0] + location_[0]
    y_0 = -(patch_shape[1] / 2) / scale_[1] + location_[1]
    y_1 = (patch_shape[1] / 2) / scale_[1] + location_[1]
    if x_0 < 0:
        x_0 = 0.0
    if x_1 > image_shape[0]:
        x_1 = image_shape[0]
    if y_0 < 0:
        y_0 = 0.0
    if y_1 > image_shape[1]:
        y_1 = image_shape[1]
    im_2[x_0:x_1, y_0:y_1, 0] = 1

    margin_0 = 1  # + int(1 / scale_[0])
    margin_1 = 1  # + int(1 / scale_[1])
    inner = img[margin_0 + x_0: -margin_0 + x_1,
                margin_1 + y_0: -margin_1 + y_1, 0]

    im_2[margin_0 + x_0: -margin_0 + x_1,
         margin_1 + y_0: -margin_1 + y_1, 0] = inner

    # import ipdb; ipdb.set_trace()

    return im_2


# ims: size times x 100000
def show_patches_on_frames(ims, locations_, scales_,
                           image_shape=(100, 100), patch_shape=(16, 16)):
    hyperparameters = {}
    hyperparameters["cutoff"] = 3
    hyperparameters["batched_window"] = True
    location = T.fmatrix()
    scale = T.fmatrix()
    x = T.fvector()
    cropper = LocallySoftRectangularCropper(
        patch_shape=patch_shape,
        hyperparameters=hyperparameters,
        kernel=Gaussian())
    patch = cropper.apply(
        x.reshape((1, 1,) + image_shape),
        np.array([list(image_shape)]),
        location,
        scale)
    get_patch = theano.function([x, location, scale], patch,
                                allow_input_downcast=True)
    final_shape = (image_shape[0], image_shape[0] + patch_shape[0] + 5)
    ret = np.ones((ims.shape[0], ) + final_shape + (3,), dtype=np.float32)
    for i in range(ims.shape[0]):
        im = ims[i]
        location_ = locations_[i]
        scale_ = scales_[i]
        patch_on_frame = show_patch_on_frame(im, location_, scale_)
        ret[i, :, :image_shape[1], :] = patch_on_frame
        ret[i, -patch_shape[0]:, image_shape[1] + 5:, :] = to_rgb1(
            get_patch(im, [location_], [scale_])[0, 0])
    return ret


# frames: T x F
def visualize_attention(frames, locations, scales, prefix=''):
    results = show_patches_on_frames(frames, locations, scales)
    for i, frame in enumerate(results):
        plt.imshow(
            frame,
            interpolation='nearest')
        plt.savefig('res' + prefix + '/img_' + str(i) + '.png')
    print 'success!'
