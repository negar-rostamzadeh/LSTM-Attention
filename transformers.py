import numpy
import fuel.transformers

class PaddingShape(fuel.transformers.Transformer):
    """Like fuel.transformers.Padding but adding shapes instead of masks.
    All dimensions may vary.
    """
    def __init__(self, data_stream, shape_sources=None, shape_dtype=None,
                 **kwargs):
        if data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        super(PaddingShape, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        if shape_sources is None:
            shape_sources = self.data_stream.sources
        self.shape_sources = shape_sources
        if shape_dtype is None:
            self.shape_dtype = numpy.uint
        else:
            self.shape_dtype = shape_dtype

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.shape_sources:
                sources.append(source + '_shape')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_shapes = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.shape_sources:
                batch_with_shapes.append(source_batch)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_batch]

            padded_batch = numpy.zeros(
                (len(source_batch),) + tuple(map(max, zip(*shapes))),
                dtype=numpy.asarray(source_batch[0]).dtype)
            for i, (sample, shape) in enumerate(zip(source_batch, shapes)):
                padded_batch[(i,) + tuple(map(slice, shape))] = sample
            batch_with_shapes.append(padded_batch)

            batch_with_shapes.append(
                numpy.array(shapes, dtype=self.shape_dtype))
        return tuple(batch_with_shapes)
