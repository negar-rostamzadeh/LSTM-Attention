import numpy
import theano
from fuel.streams import DataStream
from fuel.datasets import H5PYDataset
from fuel.schemes import ShuffledScheme
from fuel.transformers import Transformer
import h5py
import numpy as np
from StringIO import StringIO
from PIL import Image
import fuel.datasets
from fuel import config
floatX = theano.config.floatX


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


class CMVv1(H5PYDataset):
    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', False)
        super(CMVv1, self).__init__(
            ("/data/lisatmp3/cooijmat/datasets/old-cluttered-mnist-video/" +
                "cluttered-mnist-video.hdf5"),
            which_sets, **kwargs)


class Preprocessor_CMV_v1(Transformer):
    def __init__(self, data_stream, **kwargs):
        super(Preprocessor_CMV_v1, self).__init__(
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


def get_cmv_v1_streams(batch_size):
    train_dataset = CMVv1(which_sets=["train"])
    valid_dataset = CMVv1(which_sets=["valid"])
    train_ind = numpy.arange(train_dataset.num_examples)
    valid_ind = numpy.arange(valid_dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(train_ind)
    rng.shuffle(valid_ind)

    train_datastream = DataStream.default_stream(
        train_dataset,
        iteration_scheme=ShuffledScheme(train_ind, batch_size))
    train_datastream = Preprocessor_CMV_v1(train_datastream)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = Preprocessor_CMV_v1(valid_datastream)

    return train_datastream, valid_datastream


class CMVv2(fuel.datasets.H5PYDataset):
    def __init__(self, path, which_set):
        file = h5py.File(path, "r")
        super(CMVv2, self).__init__(
            file, sources=tuple("videos targets".split()),
            which_sets=(which_set,), load_in_memory=True)
        # TODO: find a way to deal with `which_sets`, especially when
        # they can be discontiguous and when `subset` is provided, and
        # when all the video ranges need to be adjusted to account for this
        self.frames = np.array(file["frames"][which_set])
        file.close()

    def get_data(self, *args, **kwargs):
        video_ranges, targets = super(
            CMVv2, self).get_data(*args, **kwargs)
        videos = list(map(self.video_from_jpegs, video_ranges))
        return videos, targets

    def video_from_jpegs(self, video_range):
        frames = self.frames[video_range[0]:video_range[1]]
        video = np.array(map(self.load_frame, frames))
        return video

    def load_frame(self, jpeg):
        image = Image.open(StringIO(jpeg.tostring()))
        image = (np.array(image.getdata(), dtype=np.float32)
                 .reshape((image.size[1], image.size[0])))
        image /= 255.0
        return image


class Preprocessor_CMV_v2(Transformer):
    def __init__(self, data_stream, len, **kwargs):
        super(Preprocessor_CMV_v2, self).__init__(
            data_stream, **kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        features = [data_point[:, np.newaxis, :, :] for data_point in data[0]]
        features = np.hstack(features)
        T, B, X, Y = features.shape
        features = features.reshape(T, B, -1)
        if len == 10:
            features = features[5:15]

        features = features[::2]
        features = features.astype('float32')
        # Now the data shape should be T x B x F
        transformed_data.append(features)
        transformed_data.append(data[1])
        return transformed_data


def get_cmv_v2_64_len20_streams(batch_size):
    path = '/data/lisatmp3/cooijmat/datasets/cmv/cmv20x64x64_png.hdf5'
    train_dataset = CMVv2(path=path, which_set="train")
    valid_dataset = CMVv2(path=path, which_set="valid")
    train_ind = numpy.arange(train_dataset.num_examples)
    valid_ind = numpy.arange(valid_dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(train_ind)
    rng.shuffle(valid_ind)

    train_datastream = DataStream.default_stream(
        train_dataset,
        iteration_scheme=ShuffledScheme(train_ind, batch_size))
    train_datastream = Preprocessor_CMV_v2(train_datastream, 20)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = Preprocessor_CMV_v2(valid_datastream, 20)

    train_datastream.sources = ('features', 'targets')
    valid_datastream.sources = ('features', 'targets')

    return train_datastream, valid_datastream


def get_cmv_v2_len20_streams(batch_size):
    path = '/data/lisatmp3/cooijmat/datasets/cmv/cmv20x100x100_png.hdf5'
    train_dataset = CMVv2(path=path, which_set="train")
    valid_dataset = CMVv2(path=path, which_set="valid")
    train_ind = numpy.arange(train_dataset.num_examples)
    valid_ind = numpy.arange(valid_dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(train_ind)
    rng.shuffle(valid_ind)

    train_datastream = DataStream.default_stream(
        train_dataset,
        iteration_scheme=ShuffledScheme(train_ind, batch_size))
    train_datastream = Preprocessor_CMV_v2(train_datastream, 20)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = Preprocessor_CMV_v2(valid_datastream, 20)

    train_datastream.sources = ('features', 'targets')
    valid_datastream.sources = ('features', 'targets')

    return train_datastream, valid_datastream


def get_cmv_v2_len10_streams(batch_size):
    path = '/data/lisatmp3/cooijmat/datasets/cmv/cmv20x100x100_png.hdf5'
    train_dataset = CMVv2(path=path, which_set="train")
    valid_dataset = CMVv2(path=path, which_set="valid")
    train_ind = numpy.arange(train_dataset.num_examples)
    valid_ind = numpy.arange(valid_dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(train_ind)
    rng.shuffle(valid_ind)

    train_datastream = DataStream.default_stream(
        train_dataset,
        iteration_scheme=ShuffledScheme(train_ind, batch_size))
    train_datastream = Preprocessor_CMV_v2(train_datastream, 10)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = Preprocessor_CMV_v2(valid_datastream, 10)

    train_datastream.sources = ('features', 'targets')
    valid_datastream.sources = ('features', 'targets')

    return train_datastream, valid_datastream


class CookingDataset(fuel.datasets.H5PYDataset):
    def __init__(self, path, which_set):
        file = h5py.File(path, "r")
        super(CookingDataset, self).__init__(
            file, sources=tuple("videos targets".split()),
            which_sets=(which_set,), load_in_memory=True)
        self.frames = np.array(file["frames"][which_set])
        file.close()

    def get_data(self, *args, **kwargs):
        video_ranges, targets = super(
            CookingDataset, self).get_data(*args, **kwargs)
        videos = list(map(self.video_from_jpegs, video_ranges))
        return videos, targets

    def video_from_jpegs(self, video_range):
        frames = self.frames[video_range[0]:video_range[1]]
        video = np.array(map(self.load_frame, frames))
        return video

    def load_frame(self, jpeg):
        image = Image.open(StringIO(jpeg.tostring())).convert("RGB")
        image = (np.array(image.getdata(), dtype=np.float32)
                 .reshape((image.size[1], image.size[0], 3)))
        image /= 255.0
        return image


class Preprocessor_Cooking(Transformer):
    def __init__(self, data_stream, **kwargs):
        super(Preprocessor_Cooking, self).__init__(
            data_stream, **kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        features = [data_point[:, np.newaxis, :, :] for data_point in data[0]]
        features = np.hstack(features)
        T, B, X, Y = features.shape
        features = features.reshape(T, B, -1)
        # features = features[5:15]
        features = features[::2]
        features = features.astype('float32')
        # Now the data shape should be T x B x F
        transformed_data.append(features)
        transformed_data.append(data[1])
        return transformed_data


def get_cooking_streams(batch_size):
    path = '/u/pezeshki/tmp3/LSTM-Attention/Cooking2_JPEG_HDF5.hdf5'
    train_dataset = CookingDataset(path=path, which_set="train")
    valid_dataset = CookingDataset(path=path, which_set="val")
    test_dataset = CookingDataset(path=path, which_set="test")
    train_ind = numpy.arange(train_dataset.num_examples)
    valid_ind = numpy.arange(valid_dataset.num_examples)
    test_ind = numpy.arange(test_dataset.num_examples)
    rng = numpy.random.RandomState(seed=1)
    rng.shuffle(train_ind)
    rng.shuffle(valid_ind)
    rng.shuffle(test_ind)

    train_datastream = DataStream.default_stream(
        train_dataset,
        iteration_scheme=ShuffledScheme(train_ind, batch_size))
    import ipdb; ipdb.set_trace()
    train_datastream = Preprocessor_Cooking(train_datastream)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = Preprocessor_Cooking(valid_datastream)

    test_datastream = DataStream.default_stream(
        test_dataset,
        iteration_scheme=ShuffledScheme(test_ind, batch_size))
    test_datastream = Preprocessor_Cooking(test_datastream)

    train_datastream.sources = ('features', 'targets')
    valid_datastream.sources = ('features', 'targets')
    test_datastream.sources = ('features', 'targets')

    return train_datastream, valid_datastream


class JpegHDF5Transformer(Transformer):
    """
    Decode jpeg and perform spatial crop if needed
    if input_size == crop_size, no crop is done
    input_size: spatially resize the input to that size
    crop_size: take a crop of that size out of the inputs
    nb_channels: number of channels of the inputs
    flip: in 'random', 'flip', 'noflip' activate flips data augmentation
    swap_rgb: Swap rgb pixel using in [2 1 0] order
    crop_type: random, corners or center type of cropping
    scale: pixel values are scale into the range [0, scale]
    nb_frames: maximum number of frame (will be zero padded)
    """
    def __init__(self,
                 input_size=(240, 320),
                 crop_size=(224, 224),
                 nchannels=3,
                 flip='random',
                 resize=True,
                 mean=None,
                 swap_rgb=False,
                 crop_type='random',
                 scale=1.,
                 nb_frames= 25,
                 *args, **kwargs):

        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = np.random.RandomState(config.default_seed)
        super(JpegHDF5Transformer, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.crop_size = crop_size
        self.nchannels = nchannels
        self.swap_rgb = swap_rgb
        self.flip = flip
        self.nb_frames = nb_frames
        self.resize = resize
        self.scale = scale
        self.mean = mean
        self.data_sources = ('targets', 'images')

        # multi-scale
        self.scales = [256, 224, 192, 168]

        # Crop coordinate
        self.crop_type = crop_type
        self.centers = np.array(input_size) / 2.0
        self.centers_crop = (self.centers[0] - self.crop_size[0] / 2.0,
                             self.centers[1] - self.crop_size[1] / 2.0)
        self.corners = []
        self.corners.append((0, 0))
        self.corners.append((0, self.input_size[1] - self.crop_size[1]))
        self.corners.append((self.input_size[0] - self.crop_size[0], 0))
        self.corners.append((self.input_size[0] - self.crop_size[0],
                             self.input_size[1] - self.crop_size[1]))
        self.corners.append(self.centers_crop)

        # Value checks
        assert self.crop_type in ['center', 'corners', 'random',
                                  'upleft', 'downleft',
                                  'upright', 'downright',
                                  'random_multiscale',
                                  'corners_multiscale']

        assert self.flip in ['random', 'flip', 'noflip']
        assert self.crop_size[0] <= self.input_size[0]
        assert self.crop_size[1] <= self.input_size[1]
        assert self.nchannels >= 1

    def multiscale_crop(self):
        scale_x = self.rng.randint(0, len(self.scales))
        scale_y = self.rng.randint(0, len(self.scales))
        crop_size = (self.scales[scale_x], self.scales[scale_y])

        centers_crop = (self.centers[0] - crop_size[0] / 2.0,
                        self.centers[1] - crop_size[1] / 2.0)
        corners = []
        corners.append((0, 0))
        corners.append((0, self.input_size[1] - crop_size[1]))
        corners.append((self.input_size[0] - crop_size[0], 0))
        corners.append((self.input_size[0] - crop_size[0],
                        self.input_size[1] - crop_size[1]))
        corners.append(centers_crop)
        return corners, crop_size

    def get_crop_coord(self, crop_size, corners):
        x_start = 0
        y_start = 0

        corner_rng = self.rng.randint(0, 5)
        if ((self.crop_type == 'random' or
             self.crop_type == 'random_multiscale')):
            if crop_size[0] <= self.input_size[0]:
                if crop_size[0] == self.input_size[0]:
                    x_start = 0
                else:
                    x_start = self.rng.randint(
                        0, self.input_size[0] - crop_size[0])
                if crop_size[1] == self.input_size[1]:
                    y_start = 0
                else:
                    y_start = self.rng.randint(
                        0, self.input_size[1] - crop_size[1])
        elif ((self.crop_type == 'corners' or
               self.crop_type == 'corners_multiscale')):
            x_start = corners[corner_rng][0]
            y_start = corners[corner_rng][1]
        elif self.crop_type == 'upleft':
            x_start = corners[0][0]
            y_start = corners[0][1]
        elif self.crop_type == 'upright':
            x_start = corners[1][0]
            y_start = corners[1][1]
        elif self.crop_type == 'downleft':
            x_start = corners[2][0]
            y_start = corners[2][1]
        elif self.crop_type == 'downright':
            x_start = corners[3][0]
            y_start = corners[3][1]
        elif self.crop_type == 'center':
            x_start = corners[4][0]
            y_start = corners[4][1]
        else:
            raise ValueError
        return x_start, y_start

    def crop(self):
        if ((self.crop_type == 'random_multiscale' or
             self.crop_type == 'corners_multiscale')):
            corners, crop_size = self.multiscale_crop()
        else:
            corners, crop_size = self.corners, self.crop_size

        x_start, y_start = self.get_crop_coord(crop_size, corners)
        bbox = (int(y_start), int(x_start),
                int(y_start + crop_size[1]), int(x_start + crop_size[0]))
        return bbox

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        batch = next(self.child_epoch_iterator)
        images, labels = self.preprocess_data(batch)
        return images, labels

    def preprocess_data(self, batch):
        # in batch[0] are all the vidis. They are the same for each fpv element
        # in batch[1] are all the frames. A group of fpv is one video

        fpv = self.nb_frames

        data_array = batch[0]
        num_videos = int(len(data_array) / fpv)
        x = np.zeros((num_videos, fpv,
                      self.crop_size[0], self.crop_size[1], self.nchannels),
                     dtype='float32')
        y = np.empty(num_videos, dtype='int64')
        for i in xrange(num_videos):
            y[i] = batch[1][i * fpv]
            do_flip = self.rng.rand(1)[0]
            bbox = self.crop()

            for j in xrange(fpv):
                data = data_array[i * fpv + j]
                # this data was stored in uint8
                data = StringIO(data.tostring())
                data.seek(0)
                img = Image.open(data)
                if (img.size[0] != self.input_size[1] and
                        img.size[1] != self.input_size[0]):
                    img = img.resize((int(self.input_size[1]),
                                      int(self.input_size[0])),
                                     Image.ANTIALIAS)
                img = img.crop(bbox)
                img = img.resize((int(self.crop_size[1]),
                                  int(self.crop_size[0])),
                                 Image.ANTIALIAS)
                # cv2.imshow('img', np.array(img))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                img = (np.array(img).astype(np.float32) / 255.0) * self.scale

                if self.nchannels == 1:
                    img = img[:, :, None]
                if self.swap_rgb and self.nchannels == 3:
                    img = img[:, :, [2, 1, 0]]
                x[i, j, :, :, :] = img[:, :, :]

                # Flip
                if self.flip == 'flip' or (self.flip == 'random'
                                           and do_flip > 0.5):
                    new_image = np.empty_like(x[i, j, :, :, :])
                    for c in xrange(self.nchannels):
                        new_image[:, :, c] = np.fliplr(x[i, j, :, :, c])
                    x[i, j, :, :, :] = new_image
        return (x, y)

# tds, vds = get_cmv_v2_streams(60)
# data = tds.get_epoch_iterator().next()
# import ipdb; ipdb.set_trace()
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# frame_5 = data[0][5, 0, :].reshape(64, 64)
# frame_7 = data[0][7, 0, :].reshape(64, 64)
# plt.imshow(frame_5, cmap=plt.gray(), interpolation='nearest')
# plt.savefig('png64_5.png')
# plt.imshow(frame_7, cmap=plt.gray(), interpolation='nearest')
# plt.savefig('png64_7.png')
# dst, dsv = get_cooking_streams(100)
