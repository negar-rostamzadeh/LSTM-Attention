import os

import numpy as np

from blocks.extensions import SimpleExtension, Printing

import util

def dump_model_parameters(file, model):
    np.savez(file,
             **dict((key, value.get_value())
                    for key, value in model.get_parameter_dict().iteritems()))

def load_model_parameters(file, model):
    parameters = np.load(file)
    parameters = dict(("/%s" % k, v) for (k, v) in parameters.items())
    model.set_parameter_values(parameters)

class Dump(SimpleExtension):
    def __init__(self, save_path, **kwargs):
        super(Dump, self).__init__(**kwargs)
        self.save_path = save_path

    def do(self, which_callback, *args, **kwargs):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        filename = "params_%i.npz" % self.main_loop.status["epochs_done"]
        dump_model_parameters(os.path.join(self.save_path, filename),
                              self.main_loop.model)

class PrintingTo(Printing):
    def __init__(self, path, **kwargs):
        super(PrintingTo, self).__init__(**kwargs)
        self.path = path
        with open(self.path, "w") as f:
            f.truncate(0)

    def do(self, *args, **kwargs):
        with util.StdoutLines() as lines:
            super(PrintingTo, self).do(*args, **kwargs)
        with open(self.path, "a") as f:
            f.write("\n".join(lines))
            f.write("\n")

class DumpMinimum(SimpleExtension):
    def __init__(self, save_path, channel_name, sign=1, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(DumpMinimum, self).__init__(**kwargs)
        self.save_path = save_path
        self.channel_name = channel_name
        self.sign = sign
        self.record_value = np.float32("inf")

    def do(self, which_callback, *args, **kwargs):
        current_value = self.main_loop.log.current_row.get(self.channel_name)
        if current_value is None:
            return
        if self.sign*current_value < self.sign*self.record_value:
            self.record_value = current_value
            self.do_dump()

    def do_dump(self):
        dump_model_parameters(self.save_path, self.main_loop.model)
