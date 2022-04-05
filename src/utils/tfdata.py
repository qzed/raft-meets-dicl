import pandas as pd
import numpy as np

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.compat.proto import summary_pb2, types_pb2


def tf_proto_tensor_to_np(tensor):
    # load flattened representation (row-major)
    if tensor.dtype == types_pb2.DT_FLOAT:
        values = np.array(tensor.float_val, dtype=np.single)
    elif tensor.dtype == types_pb2.DT_DOUBLE:
        values = np.array(tensor.double_val, dtype=np.double)
    else:
        raise NotImplementedError()     # TODO

    # reshape
    dims = tensor.tensor_shape.dim
    if len(dims) == 0:
        return values.item()

    raise NotImplementedError()         # TODO


def tfdata_scalars_to_pandas(file, tags=None):
    records = []

    for event in EventFileLoader(file).Load():
        if not event.HasField('summary'):
            continue

        for value in event.summary.value:
            # pre-filter by tags to speed things up
            if tags is not None and value.tag not in tags:
                continue

            if value.metadata.data_class == summary_pb2.DataClass.DATA_CLASS_SCALAR:
                entry = {
                    'tag': value.tag,
                    'step': event.step,
                    'time': event.wall_time,
                    'value': tf_proto_tensor_to_np(value.tensor)
                }
                records.append(entry)

    return pd.DataFrame.from_records(records)

# TODO: allow loading directories and multiple files
