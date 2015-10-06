import os
import h5py
import numpy as np
import fuel.datasets, fuel.transformers

from StringIO import StringIO
import PIL.Image as Image

import tasks
import transformers

class JpegHDF5Dataset(fuel.datasets.H5PYDataset):
    def __init__(self, which_set, load_in_memory=True):
        file = h5py.File(os.environ["KTH_JPEG_HDF5"], "r")
        super(JpegHDF5Dataset, self).__init__(file, which_sets=(which_set,),
                                              sources=tuple("videos targets".split()),
                                              load_in_memory=load_in_memory)
        self.frames = np.array(file["frames"][which_set])
        if load_in_memory:
            file.close()

    def get_data(self, *args, **kwargs):
        video_ranges, targets = super(JpegHDF5Dataset, self).get_data(*args, **kwargs)
        videos = list(map(self.video_from_jpegs, video_ranges))
        return videos, targets

    def video_from_jpegs(self, video_range):
        frames = self.frames[video_range[0]:video_range[1]]
        video = np.array(map(self.load_frame, frames))
        return video

    def load_frame(self, jpeg):
        image = Image.open(StringIO(jpeg.tostring()))
        image = np.array(image).astype(np.float32) / 255.0
        return image

def len_of_video((video, target)):
    return len(video)

class Task(tasks.Classification):
    name = "kth"

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_channels = 1
        self.n_classes = 6

    def load_datasets(self):
        return dict((which_set, JpegHDF5Dataset(which_set=which_set))
                    for which_set in "train valid test".split())

    def apply_default_transformers(self, stream):
        stream = fuel.transformers.Unpack(stream)
        stream = fuel.transformers.Batch(
            stream, fuel.schemes.ConstantScheme(32 * self.batch_size))
        stream = fuel.transformers.Mapping(
            stream, mapping=fuel.transformers.SortMapping(len_of_video))
        stream = fuel.transformers.Cache(
            stream, fuel.schemes.ConstantScheme(32))
        stream = transformers.PaddingShape(
            stream, shape_sources=["videos"])
        return stream

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 300
        return super(Task, self).get_stream_num_examples(which_set, monitor)

    def compute_batch_mean(self, x, x_shape):
        # average over time first
        time = 2
        mean_frame = x.sum(axis=time, keepdims=True)
        mean_frame /= x_shape[:, np.newaxis, [time], np.newaxis, np.newaxis]
        return mean_frame.mean(axis=0, keepdims=True)

    def preprocess(self, data):
        x, x_shape, y = data
        # introduce channel axis
        x = x[:, np.newaxis, ...]
        return (x.astype(np.float32),
                x_shape.astype(np.float32),
                y.astype(np.uint8))
