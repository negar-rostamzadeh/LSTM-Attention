from fuel.datasets.mnist import MNIST
import fuel.transformers
import tasks

class Task(tasks.Classification):
    name = "mnist"

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.n_classes = 10
        self.n_channels = 1

    def load_datasets(self):
        return dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))

    def get_stream_num_examples(self, which_set, monitor):
        if monitor and which_set == "train":
            return 10000
        return super(Task, self).get_stream_num_examples(which_set, monitor)

    def preprocess(self, data):
        x, y = data
        # remove bogus singleton dimension
        y = y.flatten()
        x_shape = np.tile([x.shape[2:]], (x.shape[0], 1))
        return (x.astype(np.float32),
                x_shape.astype(np.float32),
                y.astype(np.uint8))
