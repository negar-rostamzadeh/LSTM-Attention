"""Parser for bvlc caffe two-stream."""
# Authors:
# License: BSD 3 Clause

from sklearn.externals import joblib
from ...datasets import get_dataset_dir, download
from .caffemodel import _parse_caffe_model, parse_caffe_model, _compile_caffe_protobuf
import os
from ...utils import check_tensor, get_minibatch_indices
from .googlenet_class_labels import get_googlenet_class_label
from .googlenet_layer_names import get_googlenet_layer_names
from sklearn.base import BaseEstimator, TransformerMixin
import theano
import numpy as np
import Image

MODEL_PATH = get_dataset_dir("caffe/action_recognition")


def fetch_protobuffer_file(caffemodel_file=None,
                           mode = 'flow',
                           split="split1"):
    """Checks for existence of caffemodel protobuffer.
    Downloads it if it cannot be found."""

    if mode == 'flow':
        default_filename = os.path.join(MODEL_PATH,
                                        "cuhk_action_temporal_vgg_16_" + split +".caffemodel")
    else:
        default_filename = os.path.join(MODEL_PATH,
                                        "cuhk_action_spatial_vgg_16_" + split +".caffemodel")

    if caffemodel_file is not None:
        if os.path.exists(caffemodel_file):
            return caffemodel_file
        else:
            if os.path.exists(default_filename):
                import warnings
                warnings.warn('Did not find %s, but found and returned %s.' %
                              (caffemodel_file, default_filename))
                return default_filename
    else:
        if os.path.exists(default_filename):
            return default_filename
    # We didn't find the file, let's download it. To the specified location
    # if specified, otherwise to the default place
    if caffemodel_file is None:
        caffemodel_file = default_filename
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

    url = "http://mmlab.siat.ac.cn/very_deep_two_stream_model/"
    if mode == 'flow':
        url += "cuhk_action_temporal_vgg_16_" + split +".caffemodel"
    else:
        url += "cuhk_action_spatial_vgg_16_" + split +".caffemodel"
    download(url, caffemodel_file, progress_update_percentage=1)
    return caffemodel_file


def fetch_twostream_architecture(caffemodel_parsed=None,
                                 caffemodel_protobuffer=None,
                                 mode = 'flow',
                                 split="split1"):
    """Fetch a pickled version of the caffe model, represented as list of
    dictionaries."""

    if mode == 'flow':
        model_name = "cuhk_action_temporal_vgg_16_" + split
    else:
        model_name = "cuhk_action_spatial_vgg_16_" + split

    default_filename = os.path.join(MODEL_PATH, model_name + '.pickle')
    print default_filename
    if caffemodel_parsed is not None:
        if os.path.exists(caffemodel_parsed):
            return joblib.load(caffemodel_parsed)
        else:
            if os.path.exists(default_filename):
                import warnings
                warnings.warn('Did not find %s, but found %s. Loading it.' %
                              (caffemodel_parsed, default_filename))
                return joblib.load(default_filename)
    else:
        if os.path.exists(default_filename):
            return joblib.load(default_filename)

    # We didn't find the file: let's create it by parsing the protobuffer
    protobuf_file = fetch_protobuffer_file(caffemodel_protobuffer, split=split, mode=mode)
    model = _parse_caffe_model(protobuf_file, model_type="vgg_flows")

    if caffemodel_parsed is not None:
        joblib.dump(model, caffemodel_parsed)
    else:
        joblib.dump(model, default_filename)

    return model


def create_theano_expressions(split='split1', model=None, verbose=0,
                              selected_layers = None,
                              mode='flow',
                              inputs = None):


    ## test compile protobuf
    url = ('https://raw.githubusercontent.com/'
           'yjxiong/caffe/action_recog/src/caffe/proto/caffe.proto')
    _compile_caffe_protobuf(python_out_dir="/home/ballasn/project/sklearn-theano/sklearn_theano/models/vgg_flows", url=url)

    if model is None:
        model = fetch_twostream_architecture(split=split, mode=mode)

    layers, blobs, inputs_, param = parse_caffe_model(model, verbose=verbose,
                                                      selected_layers = selected_layers,
                                                      inputs_var = inputs)

    if inputs == None:
        data_input = inputs_['data']
    else:
        data_input = inputs[1]
    return blobs, data_input, param


