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
    def __init__(self, data_stream, **kwargs):
        super(Preprocessor_CMV_v2, self).__init__(
            data_stream, **kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        features = [data_point[:, np.newaxis, :, :] for data_point in data[0]]
        features = np.hstack(features)
        T, B, X, Y = features.shape
        features = features.reshape(T, B, -1)
        features = features.astype('float32')
        # Now the data shape should be T x B x F
        transformed_data.append(features)
        transformed_data.append(data[1])
        return transformed_data


def get_cmv_v2_streams(batch_size):
    path = '/data/lisatmp3/cooijmat/datasets/cmv/cmv20x64x64_jpeg.hdf5'
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
    train_datastream = Preprocessor_CMV_v2(train_datastream)

    valid_datastream = DataStream.default_stream(
        valid_dataset,
        iteration_scheme=ShuffledScheme(valid_ind, batch_size))
    valid_datastream = Preprocessor_CMV_v2(valid_datastream)

    train_datastream.sources = ('features', 'targets')
    valid_datastream.sources = ('features', 'targets')

    return train_datastream, valid_datastream


def aargh(data):
    fc, conv, targets = data
    # c, t -> t, c
    fc = [np.rollaxis(x, 1, 0) for x in fc]
    # c, t, y, z -> t, c, y, z
    conv = [np.rollaxis(x, 1, 0) for x in conv]
    return fc, conv, targets

from fuel.transformers import Mapping
def get_featurelevel_ucf101_streams(batch_size):
    streams = []
    for which_set in "train test".split():
        dataset = FeaturelevelUCF101Dataset(which_sets=[which_set])
        indices = numpy.arange(dataset.num_examples)
        rng = numpy.random.RandomState(seed=1)
        rng.shuffle(indices)
        datastream = DataStream.default_stream(
            dataset,
            iteration_scheme=ShuffledScheme(indices, batch_size))
        datastream.sources = tuple("fc conv targets".split())
        datastream = Mapping(datastream, aargh)
        datastream = PaddingLength(datastream, shape_sources="fc conv".split())
        streams.append(datastream)
    return streams

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

import os, cPickle, zlib, functools
max_duration = 100
def bound_duration(sources, augment=False):
    duration = sources[0].shape[1]
    crop_duration = duration
    if augment:
        crop_duration = duration / 2
    crop_duration = min(crop_duration, max_duration)
    if duration > crop_duration:
        sources = list(sources)
        # take a random chronological subsample of frames
        frames_kept = np.random.choice(duration, crop_duration, replace=False)
        frames_kept.sort()
        for i, source in enumerate(sources):
            sources[i] = source[:, frames_kept, ...]
    return sources

class FeaturelevelUCF101Dataset(fuel.datasets.H5PYDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("load_in_memory", True)
        path = os.environ["FEATURELEVEL_UCF101_HDF5"]
        super(FeaturelevelUCF101Dataset, self).__init__(path, *args, **kwargs)

    def get_data(self, *args, **kwargs):
        sources = list(super(FeaturelevelUCF101Dataset, self).get_data(*args, **kwargs))
        for i in range(2):
            sources[i] = list(map(cPickle.load, map(StringIO, map(zlib.decompress, sources[i]))))
            # move channel axis before time axis
            sources[i] = [np.rollaxis(x, 1, 0) for x in sources[i]]
        sources[:2] = list(zip(*list(map(
            functools.partial(bound_duration,
                              augment=self.which_sets[0] == "train"),
            zip(*sources[:2])))))
        # so i accidentally mixed up the two when generating the dataset
        sources[0], sources[1] = sources[1], sources[0]
        # flatten the degenerate spatial dimensions on the fc features
        sources[0] = [np.reshape(x, (x.shape[0], -1))
                      for x in sources[0]]
        # so targets are 1-based -_-
        sources[2] -= 1
        return sources

class PaddingLength(fuel.transformers.Transformer):
    """
    Like fuel.transformers.Padding but adding first-dimension
    lengths instead of masks.
    """
    def __init__(self, data_stream, shape_sources=None, shape_dtype=None,
                 **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        super(PaddingLength, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        if shape_sources is None:
            shape_sources = self.data_stream.sources
        self.shape_sources = shape_sources
        if shape_dtype is None:
            self.shape_dtype = numpy.int
        else:
            self.shape_dtype = shape_dtype

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.shape_sources:
                sources.append(source + '_length')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_shapes = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.shape_sources:
                batch_with_shapes.append(source_batch)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            fixed_shape = shapes[0][1:]

            padded_batch = numpy.zeros(
                (len(source_batch), max(lengths)) + fixed_shape,
                dtype=numpy.asarray(source_batch[0]).dtype)
            for i, (sample, length) in enumerate(zip(source_batch, lengths)):
                padded_batch[i, slice(length)] = sample
            batch_with_shapes.append(padded_batch)

            batch_with_shapes.append(
                numpy.array(lengths, dtype=self.shape_dtype))
        return tuple(batch_with_shapes)
